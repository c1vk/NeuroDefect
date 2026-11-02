from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.feature_map import display_name
from src.data.imbalance import resolve_smote
from src.data.loader import describe_dataset_path, list_datasets, load_dataset
from src.data.quality import build_quality_report
from src.pipeline.trainer import train_pipeline
from src.ui.charts import loss_chart
from src.ui.components import empty_note, html, page_header, section
from src.utils.config import ROOT, load_config
from src.utils.exceptions import NeuroDefectError


def render() -> None:
    page_header("NeuroDefect / Model Training", "Train the system")
    config = st.session_state.get("config") or load_config()
    files = list_datasets([ROOT / "data" / "sample", ROOT / "data" / "raw"])
    labels = [str(p.relative_to(ROOT)) for p in files]
    if not labels:
        empty_note("No datasets found. Add a CSV to data/sample or data/raw.")
        return

    left, right = st.columns((1.15, 1.0), gap="small")
    with left:
        with section("Optimization Parameters", key="tr_params"):
            dataset_choice = st.selectbox(
                "Dataset Selection",
                labels,
                format_func=lambda name: describe_dataset_path(ROOT / name, ROOT),
            )
            split = st.slider(
                "Train/Test Split",
                min_value=50,
                max_value=90,
                value=int((1 - float(config.training().get("test_size", 0.3))) * 100),
                step=5,
            )
            st.caption(f"{split}% / {100 - split}%")
            ga_defaults = config.ga()
            with st.expander("Advanced"):
                c1, c2 = st.columns(2)
                pop = c1.number_input("Pop Size", 4, 80, int(ga_defaults.get("population_size", 16)))
                gens = c2.number_input("Generations", 2, 80, int(ga_defaults.get("generations", 12)))
                cx = c1.number_input("Crossover Rate", 0.0, 1.0, float(ga_defaults.get("crossover_probability", 0.8)))
                mut = c2.number_input("Mutation Rate", 0.0, 1.0, float(ga_defaults.get("mutation_probability", 0.05)))
                probe = str(ga_defaults.get("fitness_model", "logistic"))
                if probe == "bpn":
                    st.caption("Feature search is scored with a small copy of the neural network.")
                else:
                    st.caption(
                        "Feature search is a logistic filter. It does not score subsets with the neural network. "
                        "Research mode in Settings does."
                    )
                stem = Path(dataset_choice).stem
                policy_on = resolve_smote(
                    stem,
                    bool(config.smote().get("enabled", True)),
                    skip=config.smote().get("skip_for"),
                )
                if st.session_state.get("_smote_for") != dataset_choice:
                    st.session_state["_smote_for"] = dataset_choice
                    st.session_state["smote_on"] = policy_on
                smote_on = st.toggle("SMOTE Oversampling", key="smote_on")
                if not policy_on:
                    st.caption("SMOTE starts off for CM1 and JM1 because the ablation showed it hurt them. Turn it on to force it. The test set is never resampled.")
                else:
                    st.caption("SMOTE runs on the training rows only. Class weights are turned off for that run so the rows are not balanced twice.")
            train_clicked = st.button("Train Model Engine", type="primary", width="stretch")

    with right:
        with section("Training Telemetry", key="tr_telem"):
            progress_slot = st.empty()
            status_slot = st.empty()
            chart_slot = st.empty()
            m1, m2 = st.columns(2)
            acc_slot, loss_slot = m1.empty(), m2.empty()
            result = st.session_state.get("result")
            if result and not train_clicked:
                hist = result.bpn_history
                n = len(hist.train_loss)
                progress_slot.progress(1.0, text=f"Finished at epoch {n}")
                status_slot.markdown(
                    f"<p style='color:#3b82f6;font-weight:600'>Training complete · {n} epoch{'s' if n != 1 else ''}</p>",
                    unsafe_allow_html=True,
                    anchors=False,
                )
                if hist.train_loss:
                    chart_slot.plotly_chart(
                        loss_chart(hist.train_loss),
                        width="stretch",
                        config={"displayModeBar": False, "responsive": True},
                        theme=None,
                    )
                last_acc = hist.train_acc[-1] if hist.train_acc else 0
                last_loss = hist.train_loss[-1] if hist.train_loss else 0
                acc_slot.markdown(
                    f"<div class='nd-mini-metric'><div class='nd-kpi-top'>Current Accuracy</div>"
                    f"<div class='nd-kpi-value' style='color:#16a34a;font-size:1.4rem'>{last_acc:.3f}</div></div>",
                    unsafe_allow_html=True,
                    anchors=False,
                )
                loss_slot.markdown(
                    f"<div class='nd-mini-metric'><div class='nd-kpi-top'>Current Loss</div>"
                    f"<div class='nd-kpi-value' style='color:#ef4444;font-size:1.4rem'>{last_loss:.3f}</div></div>",
                    unsafe_allow_html=True,
                    anchors=False,
                )
            else:
                status_slot.caption("Telemetry appears while the engine is training.")

    if train_clicked:
        try:
            loaded = load_dataset(ROOT / dataset_choice)
            if loaded.target_column is None:
                st.error("Target column was not found. Open Dataset Analysis and select it manually.")
                return
            st.session_state.dataset = loaded
            st.session_state.quality = build_quality_report(loaded.frame, loaded.target_column, loaded.name)
            overrides = {
                "training": {"test_size": (100 - split) / 100.0},
                "ga": {
                    "population_size": int(pop),
                    "generations": int(gens),
                    "crossover_probability": float(cx),
                    "mutation_probability": float(mut),
                },
                "smote": {"enabled": bool(smote_on), "force": bool(smote_on)},
            }
            cfg = config.with_overrides(**overrides)
            st.session_state.config = cfg

            def on_progress(phase, frac, message, extra):
                progress_slot.progress(min(max(frac, 0.0), 1.0), text=message)
                status_slot.markdown(
                    f"<p style='color:#3b82f6;font-weight:600'>{message}</p>",
                    unsafe_allow_html=True,
                    anchors=False,
                )
                history = extra.get("history")
                if history is not None and history.train_loss:
                    chart_slot.plotly_chart(
                        loss_chart(history.train_loss),
                        width="stretch",
                        config={"displayModeBar": False, "responsive": True},
                        theme=None,
                    )
                    acc_slot.markdown(
                        f"<div class='nd-mini-metric'><div class='nd-kpi-top'>Current Accuracy</div>"
                        f"<div class='nd-kpi-value' style='color:#16a34a;font-size:1.4rem'>{history.train_acc[-1]:.3f}</div></div>",
                        unsafe_allow_html=True,
                        anchors=False,
                    )
                    loss_slot.markdown(
                        f"<div class='nd-mini-metric'><div class='nd-kpi-top'>Current Loss</div>"
                        f"<div class='nd-kpi-value' style='color:#ef4444;font-size:1.4rem'>{history.train_loss[-1]:.3f}</div></div>",
                        unsafe_allow_html=True,
                        anchors=False,
                    )

            with st.spinner("Training NeuroDefect — this uses the real pipeline, not cached demo scores."):
                result = train_pipeline(loaded, cfg, progress=on_progress)
            st.session_state.result = result
            st.success(f"Training finished in {result.duration:.1f}s. Artifacts saved to {result.run_dir}.")
            st.rerun()
        except NeuroDefectError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    result = st.session_state.get("result")
    with section("GA-Selected Feature Space", key="tr_features"):
        if result:
            chips = " ".join(
                f'<span class="nd-feature">✓ {display_name(name)}</span>' for name in result.selected_features
            )
            html(
                f"<p class='nd-note'>{len(result.selected_features)} of {len(result.all_features)} optimized</p>{chips}"
            )
            with st.expander("Preprocessing log"):
                st.code("\n".join(result.logs))
                st.write(
                    {
                        "Before SMOTE defective / clean": f"{result.smote.before_defective} / {result.smote.before_clean}",
                        "After SMOTE defective / clean": f"{result.smote.after_defective} / {result.smote.after_clean}",
                        "GA fitness": round(result.ga_fitness, 4),
                        "GA runtime (s)": round(result.ga_runtime, 2),
                        "Backend": result.bpn_history.backend,
                    }
                )
        else:
            st.caption("Selected features will appear here after a successful training run.")

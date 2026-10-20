const pptxgen = require("pptxgenjs");

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "NeuroDefect — Project Presentation";
  pres.author = "CH Vamsi Krishna, G Srinivas";
  pres.subject =
    "A Hybrid Neuro-Fuzzy-GA Decision Support System for Software Defect Prediction and Bug-Severity Assessment";

  const C = {
    navy: "1B2A4A",
    dk: "0F1B33",
    slate: "2D3E50",
    teal: "0E7C86",
    gold: "D4A843",
    off: "FAFAF7",
    wg: "E8E4DD",
    tx: "2C2C2C",
    mu: "6B7280",
    wh: "FFFFFF",
    lt: "E6F4F5",
    red: "B03A2E",
    grn: "1D7A4E",
    dkgold: "A07D2E",
    card: "FFFFFF",
  };
  const H = "Century Gothic";
  const B = "Calibri";
  const W = 13.3;
  const HT = 7.5;

  const goldBar = (s) =>
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0,
      y: 0,
      w: W,
      h: 0.07,
      fill: { color: C.gold },
    });

  const footer = (s, page, total = 13) => {
    s.addText("NeuroDefect  ·  Academic Project Presentation", {
      x: 0.55,
      y: 7.18,
      w: 9.5,
      h: 0.2,
      fontFace: B,
      fontSize: 11,
      color: C.mu,
      margin: 0,
    });
    s.addText(String(page).padStart(2, "0") + "  /  " + String(total).padStart(2, "0"), {
      x: 11.4,
      y: 7.18,
      w: 1.35,
      h: 0.2,
      fontFace: B,
      fontSize: 11,
      color: C.mu,
      align: "right",
      margin: 0,
    });
  };

  const sectionKicker = (s, text) =>
    s.addText(text, {
      x: 0.55,
      y: 0.28,
      w: 12.2,
      h: 0.28,
      fontFace: B,
      fontSize: 12,
      color: C.teal,
      bold: true,
      charSpacing: 1.5,
      margin: 0,
    });

  const title = (s, text, y = 0.55) =>
    s.addText(text, {
      x: 0.55,
      y,
      w: 12.2,
      h: 0.55,
      fontFace: H,
      fontSize: 28,
      color: C.navy,
      bold: true,
      margin: 0,
    });

  // ─────────────────────────────────────────────
  // 1. TITLE
  // ─────────────────────────────────────────────
  let s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "ACADEMIC PROJECT PRESENTATION");
  s.addText("NeuroDefect", {
    x: 0.55,
    y: 0.7,
    w: 12.2,
    h: 0.95,
    fontFace: H,
    fontSize: 48,
    color: C.navy,
    bold: true,
    margin: 0,
  });

  s.addText(
    "A Hybrid Neuro-Fuzzy-GA Decision Support System for\nSoftware Defect Prediction and Bug-Severity Assessment",
    {
      x: 0.55,
      y: 1.7,
      w: 12.2,
      h: 0.95,
      fontFace: B,
      fontSize: 20,
      color: C.teal,
      italic: true,
      margin: 0,
    }
  );

  s.addText("TEAM MEMBERS", {
    x: 0.55,
    y: 2.9,
    w: 12.2,
    h: 0.28,
    fontFace: B,
    fontSize: 12,
    color: C.teal,
    bold: true,
    charSpacing: 1.5,
    margin: 0,
  });

  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.55,
    y: 3.35,
    w: 5.95,
    h: 2.15,
    fill: { color: C.card },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.8,
    y: 3.35,
    w: 5.95,
    h: 2.15,
    fill: { color: C.card },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.55,
    y: 3.35,
    w: 0.1,
    h: 2.15,
    fill: { color: C.gold },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 6.8,
    y: 3.35,
    w: 0.1,
    h: 2.15,
    fill: { color: C.gold },
  });

  s.addText("CH Vamsi Krishna", {
    x: 0.95,
    y: 3.7,
    w: 5.3,
    h: 0.5,
    fontFace: H,
    fontSize: 24,
    color: C.navy,
    bold: true,
    margin: 0,
  });
  s.addText("24BCE8195", {
    x: 0.95,
    y: 4.3,
    w: 5.3,
    h: 0.45,
    fontFace: B,
    fontSize: 18,
    color: C.teal,
    margin: 0,
  });

  s.addText("G Srinivas", {
    x: 7.2,
    y: 3.7,
    w: 5.3,
    h: 0.5,
    fontFace: H,
    fontSize: 24,
    color: C.navy,
    bold: true,
    margin: 0,
  });
  s.addText("24BCE8126", {
    x: 7.2,
    y: 4.3,
    w: 5.3,
    h: 0.45,
    fontFace: B,
    fontSize: 18,
    color: C.teal,
    margin: 0,
  });

  s.addText(
    "Software defect prediction  ·  Explainable AI  ·  Genetic algorithms  ·  Fuzzy risk assessment",
    {
      x: 0.55,
      y: 5.75,
      w: 12.2,
      h: 0.4,
      fontFace: B,
      fontSize: 14,
      color: C.mu,
      margin: 0,
    }
  );
  footer(s, 1);

  // ─────────────────────────────────────────────
  // 2. AGENDA
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "OVERVIEW");
  title(s, "What this presentation covers");

  const agenda = [
    ["01", "What we are building", "An explainable decision-support system for software defect prediction."],
    ["02", "Problem statement", "Why late, black-box defect detection fails testing teams."],
    ["03", "Why this project", "The research gap we chose, and what faculty can inspect."],
    ["04", "Datasets & sources", "NASA PROMISE / MDP projects with official reference links."],
    ["05", "Method & tech stack", "GA feature selection, neural predictor, fuzzy risk, Streamlit UI."],
    ["06", "Integrity notes", "What the system claims — and what it must not claim."],
  ];
  agenda.forEach((item, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.55 + col * 6.35;
    const y = 1.4 + row * 1.75;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 6.05,
      h: 1.55,
      fill: { color: C.card },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
      rectRadius: 0.06,
    });
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.22,
      y: y + 0.42,
      w: 0.7,
      h: 0.7,
      fill: { color: C.navy },
    });
    s.addText(item[0], {
      x: x + 0.22,
      y: y + 0.55,
      w: 0.7,
      h: 0.45,
      fontFace: H,
      fontSize: 14,
      color: C.gold,
      bold: true,
      align: "center",
      margin: 0,
    });
    s.addText(item[1], {
      x: x + 1.12,
      y: y + 0.32,
      w: 4.65,
      h: 0.42,
      fontFace: H,
      fontSize: 18,
      color: C.navy,
      bold: true,
      margin: 0,
    });
    s.addText(item[2], {
      x: x + 1.12,
      y: y + 0.78,
      w: 4.65,
      h: 0.5,
      fontFace: B,
      fontSize: 14,
      color: C.tx,
      margin: 0,
    });
  });
  footer(s, 2);

  // ─────────────────────────────────────────────
  // 3. WHAT WE ARE DOING
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "PROJECT PURPOSE");
  title(s, "What we are doing");

  s.addText(
    "NeuroDefect is an explainable software defect prediction and decision-support system. It helps testers decide which modules to inspect first — and why.",
    {
      x: 0.55,
      y: 1.2,
      w: 12.2,
      h: 0.55,
      fontFace: B,
      fontSize: 16,
      color: C.tx,
      margin: 0,
    }
  );

  const doing = [
    {
      k: "Predict",
      t: "Is this module defective?",
      d: "A neural network trained on NASA software metrics outputs a defect probability and a defective / non-defective label.",
    },
    {
      k: "Assess",
      t: "How urgent is the risk?",
      d: "Fuzzy inference converts probability plus complexity, size, and Halstead metrics into a derived risk severity: Low, Medium, High, or Critical.",
    },
    {
      k: "Explain",
      t: "Why did the system say that?",
      d: "The UI shows selected features, fired fuzzy rules, contributing metrics, prediction confidence, and a testing recommendation.",
    },
  ];
  doing.forEach((card, i) => {
    const x = 0.55 + i * 4.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y: 1.95,
      w: 3.95,
      h: 4.55,
      fill: { color: C.navy },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y: 1.95,
      w: 3.95,
      h: 0.1,
      fill: { color: C.gold },
    });
    s.addText(card.k.toUpperCase(), {
      x: x + 0.28,
      y: 2.3,
      w: 3.4,
      h: 0.32,
      fontFace: B,
      fontSize: 13,
      color: C.gold,
      bold: true,
      charSpacing: 1.6,
      margin: 0,
    });
    s.addText(card.t, {
      x: x + 0.28,
      y: 2.75,
      w: 3.4,
      h: 1.15,
      fontFace: H,
      fontSize: 22,
      color: C.wh,
      bold: true,
      margin: 0,
    });
    s.addText(card.d, {
      x: x + 0.28,
      y: 4.1,
      w: 3.4,
      h: 2.0,
      fontFace: B,
      fontSize: 15,
      color: C.wg,
      margin: 0,
    });
  });
  footer(s, 3);

  // ─────────────────────────────────────────────
  // 4. PROBLEM STATEMENT
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "PROBLEM STATEMENT");
  title(s, "Software defects are expensive. Finding them is hard.");

  const problems = [
    {
      n: "01",
      h: "Defects found late cost more",
      b: "A faulty module that escapes into later phases is far more expensive to fix than one caught during targeted review.",
    },
    {
      n: "02",
      h: "Testing time is limited",
      b: "Teams cannot inspect every module. They need a ranked testing priority, not a single accuracy number.",
    },
    {
      n: "03",
      h: "Black-box models do not help reviewers",
      b: "A classifier that only says “defective” without reasons cannot support a code-review decision.",
    },
    {
      n: "04",
      h: "NASA labels are binary, not severity",
      b: "CM1, JM1 and PC5 mark defective vs clean. They do not provide Low / Medium / High / Critical ground-truth bug severity.",
    },
  ];
  problems.forEach((p, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.55 + col * 6.35;
    const y = 1.4 + row * 2.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 6.05,
      h: 2.35,
      fill: { color: C.card },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
      rectRadius: 0.06,
    });
    s.addText(p.n, {
      x: x + 0.3,
      y: y + 0.28,
      w: 1.1,
      h: 0.45,
      fontFace: H,
      fontSize: 20,
      color: C.gold,
      bold: true,
      margin: 0,
    });
    s.addText(p.h, {
      x: x + 1.4,
      y: y + 0.32,
      w: 4.35,
      h: 0.45,
      fontFace: H,
      fontSize: 18,
      color: C.navy,
      bold: true,
      margin: 0,
    });
    s.addText(p.b, {
      x: x + 0.3,
      y: y + 1.0,
      w: 5.45,
      h: 1.05,
      fontFace: B,
      fontSize: 15,
      color: C.tx,
      margin: 0,
    });
  });
  footer(s, 4);

  // ─────────────────────────────────────────────
  // 5. WHY THIS PROJECT
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "MOTIVATION");
  title(s, "Why we chose this project");

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.55,
    y: 1.3,
    w: 12.2,
    h: 1.35,
    fill: { color: C.navy },
    rectRadius: 0.06,
  });
  s.addText(
    "We wanted a research prototype a faculty member can interrogate: “Where did this prediction come from?” — not a demo that only looks accurate.",
    {
      x: 0.85,
      y: 1.55,
      w: 11.6,
      h: 0.9,
      fontFace: B,
      fontSize: 18,
      color: C.wh,
      margin: 0,
    }
  );

  const why = [
    {
      h: "A real software-engineering need",
      b: "Defect prediction is a standard research problem, but testers need ranking, risk, and explanation — not accuracy alone.",
    },
    {
      h: "A hybrid that is actually hybrid",
      b: "Genetic algorithms pick a compact feature subset. A neural net estimates defect probability. Fuzzy rules turn that probability plus code metrics into an interpretable risk band.",
    },
    {
      h: "Honesty about NASA data",
      b: "We treat NASA labels as binary defect flags. Risk severity is derived. We do not present it as experimentally validated bug severity.",
    },
    {
      h: "Reproducible science",
      b: "Leakage-safe train/test split, SMOTE on training data only, saved artifacts, and tests. Every number on the dashboard must come from a trained model.",
    },
  ];
  why.forEach((item, i) => {
    const x = 0.55 + (i % 2) * 6.35;
    const y = 2.9 + Math.floor(i / 2) * 1.9;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 6.05,
      h: 1.7,
      fill: { color: C.card },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
      rectRadius: 0.06,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y,
      w: 0.1,
      h: 1.7,
      fill: { color: i % 2 === 0 ? C.teal : C.gold },
    });
    s.addText(item.h, {
      x: x + 0.35,
      y: y + 0.22,
      w: 5.5,
      h: 0.4,
      fontFace: H,
      fontSize: 16,
      color: C.navy,
      bold: true,
      margin: 0,
    });
    s.addText(item.b, {
      x: x + 0.35,
      y: y + 0.68,
      w: 5.5,
      h: 0.8,
      fontFace: B,
      fontSize: 14,
      color: C.tx,
      margin: 0,
    });
  });
  footer(s, 5);

  // ─────────────────────────────────────────────
  // 6. DATASETS
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "DATA");
  title(s, "NASA PROMISE / MDP defect datasets");

  s.addText(
    "Public software-metric datasets originally released by NASA’s Metrics Data Program and redistributed through the PROMISE Software Engineering Repository.",
    {
      x: 0.55,
      y: 1.18,
      w: 12.2,
      h: 0.4,
      fontFace: B,
      fontSize: 15,
      color: C.tx,
      margin: 0,
    }
  );

  const tableHeader = [
    {
      text: "Dataset",
      options: { fill: { color: C.navy }, color: C.gold, bold: true, align: "left" },
    },
    {
      text: "Project",
      options: { fill: { color: C.navy }, color: C.gold, bold: true, align: "left" },
    },
    {
      text: "Language",
      options: { fill: { color: C.navy }, color: C.gold, bold: true, align: "center" },
    },
    {
      text: "Modules",
      options: { fill: { color: C.navy }, color: C.gold, bold: true, align: "center" },
    },
    {
      text: "Metrics",
      options: { fill: { color: C.navy }, color: C.gold, bold: true, align: "center" },
    },
    {
      text: "Defect rate",
      options: { fill: { color: C.navy }, color: C.gold, bold: true, align: "center" },
    },
  ];
  const rowStyle = (vals, alt) =>
    vals.map((v, idx) => ({
      text: v,
      options: {
        fill: { color: alt ? "F4F1EA" : C.wh },
        color: C.tx,
        align: idx >= 2 ? "center" : "left",
        bold: idx === 0,
      },
    }));

  s.addTable(
    [
      tableHeader,
      rowStyle(["CM1", "NASA spacecraft instrument", "C", "498", "21", "9.83%"], false),
      rowStyle(["JM1", "Real-time predictive ground system", "C", "10,885", "21", "19.35%"], true),
      rowStyle(["PC5", "Flight software, Earth-orbiting satellite", "C++", "17,186", "38–39", "~2.9%"], false),
    ],
    {
      x: 0.55,
      y: 1.7,
      w: 12.2,
      h: 2.35,
      colW: [1.5, 4.3, 1.5, 1.6, 1.5, 1.8],
      border: [{ pt: 0 }, { pt: 0 }, { pt: 0 }, { pt: 0 }],
      fontFace: B,
      fontSize: 14,
      color: C.tx,
      valign: "middle",
    }
  );

  s.addText(
    "Counts follow the widely cited PROMISE releases (Shirabad & Menzies, 2005). Cleaned NASA variants differ slightly — we document which file is loaded.",
    {
      x: 0.55,
      y: 4.15,
      w: 12.2,
      h: 0.32,
      fontFace: B,
      fontSize: 12,
      color: C.mu,
      italic: true,
      margin: 0,
    }
  );

  s.addText("Official and citable sources  (links are clickable)", {
    x: 0.55,
    y: 4.55,
    w: 12.2,
    h: 0.32,
    fontFace: H,
    fontSize: 16,
    color: C.navy,
    bold: true,
    margin: 0,
  });

  const links = [
    [
      "PROMISE datasets page",
      "http://promise.site.uottawa.ca/SERepository/datasets-page.html",
    ],
    ["CM1 file (PROMISE)", "http://promise.site.uottawa.ca/SERepository/datasets/cm1.arff"],
    ["JM1 file (PROMISE)", "http://promise.site.uottawa.ca/SERepository/datasets/jm1.arff"],
    ["tera-PROMISE / NASA defect repo", "http://openscience.us/repo/defect/"],
    ["Shepperd et al. 2013, IEEE TSE (data quality)", "https://doi.org/10.1109/TSE.2013.11"],
    ["Cleaned NASA data (Zenodo)", "https://doi.org/10.5281/zenodo.268405"],
  ];
  links.forEach((pair, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 0.55 + col * 6.35;
    const y = 4.95 + row * 0.55;
    s.addText(pair[0] + "  →  ", {
      x,
      y,
      w: 6.1,
      h: 0.22,
      fontFace: B,
      fontSize: 12,
      color: C.navy,
      bold: true,
      margin: 0,
    });
    s.addText(pair[1], {
      x,
      y: y + 0.22,
      w: 6.1,
      h: 0.22,
      fontFace: B,
      fontSize: 11,
      color: C.teal,
      margin: 0,
      hyperlink: { url: pair[1] },
    });
  });
  footer(s, 6);

  // ─────────────────────────────────────────────
  // 7. WHAT IS IN THE DATA
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "FEATURES");
  title(s, "What each module looks like to the model");

  const feats = [
    {
      h: "McCabe complexity",
      b: "Cyclomatic complexity v(g), essential complexity, design complexity, and branch count. These describe how many independent paths a tester must cover.",
    },
    {
      h: "Halstead software science",
      b: "Length, volume, difficulty, effort, and time estimates from operator and operand counts. Dense, “effort-heavy” modules are historically defect-prone.",
    },
    {
      h: "Size & comments",
      b: "Lines of code, blank lines, comment lines. Size alone is not destiny — but large modules with high branching raise review cost.",
    },
    {
      h: "Binary defect label",
      b: "defects = true if the module had one or more faults. This is the only ground-truth target. Severity is not in the file.",
    },
  ];
  feats.forEach((f, i) => {
    const x = 0.55 + (i % 2) * 6.35;
    const y = 1.35 + Math.floor(i / 2) * 2.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 6.05,
      h: 2.35,
      fill: { color: i === 3 ? C.navy : C.card },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
      rectRadius: 0.06,
    });
    s.addText(f.h, {
      x: x + 0.35,
      y: y + 0.3,
      w: 5.4,
      h: 0.45,
      fontFace: H,
      fontSize: 18,
      color: i === 3 ? C.gold : C.navy,
      bold: true,
      margin: 0,
    });
    s.addText(f.b, {
      x: x + 0.35,
      y: y + 0.9,
      w: 5.4,
      h: 1.15,
      fontFace: B,
      fontSize: 15,
      color: i === 3 ? C.wg : C.tx,
      margin: 0,
    });
  });
  footer(s, 7);

  // ─────────────────────────────────────────────
  // 8. ARCHITECTURE
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "SYSTEM DESIGN");
  title(s, "End-to-end pipeline — leakage-safe by construction");

  const steps = [
    ["1", "Load &\nvalidate", "CSV / XLSX\ntarget check"],
    ["2", "Clean &\nsplit", "Stratified\ntrain / test"],
    ["3", "Preprocess\n+ SMOTE", "Fit on train\nonly"],
    ["4", "GA feature\nselection", "Compact\nsubset"],
    ["5", "BPN neural\npredictor", "Defect\nprobability"],
    ["6", "Fuzzy risk\nengine", "Severity +\nrules"],
  ];
  steps.forEach((st, i) => {
    const x = 0.45 + i * 2.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y: 1.45,
      w: 1.95,
      h: 2.55,
      fill: { color: C.navy },
      rectRadius: 0.08,
    });
    s.addShape(pres.shapes.OVAL, {
      x: x + 0.62,
      y: 1.62,
      w: 0.7,
      h: 0.7,
      fill: { color: C.gold },
    });
    s.addText(st[0], {
      x: x + 0.62,
      y: 1.75,
      w: 0.7,
      h: 0.45,
      fontFace: H,
      fontSize: 18,
      color: C.dk,
      bold: true,
      align: "center",
      margin: 0,
    });
    s.addText(st[1], {
      x: x + 0.1,
      y: 2.42,
      w: 1.75,
      h: 0.85,
      fontFace: H,
      fontSize: 14,
      color: C.wh,
      bold: true,
      align: "center",
      margin: 0,
    });
    s.addText(st[2], {
      x: x + 0.1,
      y: 3.3,
      w: 1.75,
      h: 0.55,
      fontFace: B,
      fontSize: 12,
      color: C.wg,
      align: "center",
      margin: 0,
    });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.RIGHT_ARROW, {
        x: x + 1.92,
        y: 2.5,
        w: 0.26,
        h: 0.22,
        fill: { color: C.gold },
      });
    }
  });

  const guards = [
    "Never fit a scaler on the full dataset before splitting",
    "Never apply SMOTE to validation or test data",
    "Never run GA feature selection on the complete dataset",
    "Never tune hyperparameters on the test set",
  ];
  s.addText("Non-negotiable data-leakage rules", {
    x: 0.55,
    y: 4.25,
    w: 12.2,
    h: 0.35,
    fontFace: H,
    fontSize: 16,
    color: C.navy,
    bold: true,
    margin: 0,
  });
  guards.forEach((g, i) => {
    const x = 0.55 + (i % 2) * 6.35;
    const y = 4.7 + Math.floor(i / 2) * 0.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 6.05,
      h: 0.72,
      fill: { color: C.lt },
      rectRadius: 0.05,
    });
    s.addText(g, {
      x: x + 0.25,
      y: y + 0.16,
      w: 5.55,
      h: 0.42,
      fontFace: B,
      fontSize: 14,
      color: C.navy,
      margin: 0,
    });
  });
  footer(s, 8);

  // ─────────────────────────────────────────────
  // 9. HYBRID METHOD
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "METHOD");
  title(s, "Three parts, one decision");

  const methods = [
    {
      tag: "GA",
      h: "Genetic algorithm\nfeature selection",
      items: [
        "Chromosome = binary mask over metrics",
        "Fitness = F1 / MCC − λ × feature ratio",
        "Avoids “use every NASA metric”",
        "Runs on training data only",
      ],
    },
    {
      tag: "NN",
      h: "BPN neural\ndefect predictor",
      items: [
        "Dense 64 → ReLU → Dropout",
        "Dense 32 → ReLU → Dropout",
        "Dense 1 → Sigmoid probability",
        "SGD with momentum, or Adam",
      ],
    },
    {
      tag: "FZ",
      h: "Fuzzy risk\nassessment engine",
      items: [
        "Inputs: complexity, LOC, volume…",
        "Plus neural defect probability",
        "~14 documented Mamdani rules",
        "Output: Low / Med / High / Critical",
      ],
    },
  ];
  methods.forEach((m, i) => {
    const x = 0.55 + i * 4.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y: 1.3,
      w: 3.95,
      h: 5.2,
      fill: { color: C.card },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y: 1.3,
      w: 3.95,
      h: 1.7,
      fill: { color: C.navy },
    });
    s.addText(m.tag, {
      x: x + 0.25,
      y: 1.42,
      w: 3.45,
      h: 0.32,
      fontFace: B,
      fontSize: 13,
      color: C.gold,
      bold: true,
      charSpacing: 2,
      margin: 0,
    });
    s.addText(m.h, {
      x: x + 0.25,
      y: 1.78,
      w: 3.45,
      h: 1.05,
      fontFace: H,
      fontSize: 20,
      color: C.wh,
      bold: true,
      margin: 0,
    });
    m.items.forEach((line, j) => {
      const iy = 3.25 + j * 0.72;
      s.addShape(pres.shapes.OVAL, {
        x: x + 0.32,
        y: iy + 0.12,
        w: 0.16,
        h: 0.16,
        fill: { color: C.gold },
      });
      s.addText(line, {
        x: x + 0.6,
        y: iy,
        w: 3.1,
        h: 0.6,
        fontFace: B,
        fontSize: 14,
        color: C.tx,
        margin: 0,
      });
    });
  });
  footer(s, 9);

  // ─────────────────────────────────────────────
  // 10. TECH STACK
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "IMPLEMENTATION");
  title(s, "Technology stack");

  const stack = [
    { g: "Language", items: "Python 3.11 / 3.12" },
    { g: "Data", items: "pandas  ·  NumPy" },
    { g: "Classical ML", items: "scikit-learn  ·  Logistic, RF, SVM, Majority baseline" },
    { g: "Neural net", items: "PyTorch BPN  (NumPy fallback if needed)" },
    { g: "Genetic algorithm", items: "DEAP" },
    { g: "Class imbalance", items: "imbalanced-learn  ·  SMOTE on train only" },
    { g: "Fuzzy logic", items: "Interpretable Mamdani engine  ·  scikit-fuzzy" },
    { g: "Dashboard", items: "Streamlit  ·  Plotly" },
    { g: "Persistence", items: "joblib  ·  PyTorch checkpoints  ·  YAML config" },
    { g: "Quality", items: "pytest  ·  Ruff / Black" },
  ];
  stack.forEach((row, i) => {
    const col = i % 2;
    const r = Math.floor(i / 2);
    const x = 0.55 + col * 6.35;
    const y = 1.3 + r * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 6.05,
      h: 0.9,
      fill: { color: C.card },
      shadow: { type: "outer", color: "000000", blur: 6, offset: 1, angle: 135, opacity: 0.07 },
      rectRadius: 0.05,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x,
      y,
      w: 0.1,
      h: 0.9,
      fill: { color: C.gold },
    });
    s.addText(row.g, {
      x: x + 0.3,
      y: y + 0.1,
      w: 5.5,
      h: 0.28,
      fontFace: H,
      fontSize: 13,
      color: C.teal,
      bold: true,
      margin: 0,
    });
    s.addText(row.items, {
      x: x + 0.3,
      y: y + 0.42,
      w: 5.5,
      h: 0.35,
      fontFace: B,
      fontSize: 15,
      color: C.navy,
      margin: 0,
    });
  });
  footer(s, 10);

  // ─────────────────────────────────────────────
  // 11. DASHBOARD
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "USER INTERFACE");
  title(s, "Six pages of the decision dashboard");

  const pages = [
    { n: "01", h: "Dashboard", b: "Modules, defect rate, selected features, model status, recent predictions." },
    { n: "02", h: "Dataset Analysis", b: "Dimensions, missingness, imbalance, descriptive stats, correlation matrix." },
    { n: "03", h: "Model Training", b: "Split, GA settings, SMOTE toggle, live telemetry, selected feature chips." },
    { n: "04", h: "Predictions", b: "Pick a module or override metrics. See probability, risk tier, rules, action." },
    { n: "05", h: "Model Comparison", b: "Baselines vs Neuro-Fuzzy-GA: F1, MCC, ROC, confusion matrix, testing budget." },
    { n: "06", h: "Explainability", b: "GA feature ranks, membership functions, fired rule base and strengths." },
  ];
  pages.forEach((p, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.55 + col * 4.2;
    const y = 1.35 + row * 2.65;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x,
      y,
      w: 3.95,
      h: 2.45,
      fill: { color: C.card },
      shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
      rectRadius: 0.06,
    });
    s.addText(p.n, {
      x: x + 0.25,
      y: y + 0.22,
      w: 3.45,
      h: 0.35,
      fontFace: H,
      fontSize: 14,
      color: C.gold,
      bold: true,
      margin: 0,
    });
    s.addText(p.h, {
      x: x + 0.25,
      y: y + 0.6,
      w: 3.45,
      h: 0.45,
      fontFace: H,
      fontSize: 18,
      color: C.navy,
      bold: true,
      margin: 0,
    });
    s.addText(p.b, {
      x: x + 0.25,
      y: y + 1.15,
      w: 3.45,
      h: 1.05,
      fontFace: B,
      fontSize: 14,
      color: C.tx,
      margin: 0,
    });
  });
  footer(s, 11);

  // ─────────────────────────────────────────────
  // 12. OUTPUTS + INTEGRITY
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "OUTPUTS  &  SCIENTIFIC INTEGRITY");
  title(s, "What the system reports — and what it does not");

  const outs = [
    "Defect probability from the trained neural model",
    "Defective / non-defective prediction",
    "Derived risk severity (Low / Medium / High / Critical)",
    "Prediction confidence (distance from 0.5, not a CI)",
    "Important risk factors and fired fuzzy rules",
    "Testing recommendation generated from those results",
  ];
  outs.forEach((line, i) => {
    const y = 1.3 + i * 0.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55,
      y,
      w: 6.5,
      h: 0.48,
      fill: { color: C.lt },
      rectRadius: 0.04,
    });
    s.addText(line, {
      x: 0.75,
      y: y + 0.08,
      w: 6.1,
      h: 0.32,
      fontFace: B,
      fontSize: 14,
      color: C.navy,
      margin: 0,
    });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.3,
    y: 1.3,
    w: 5.45,
    h: 5.15,
    fill: { color: C.navy },
    rectRadius: 0.08,
  });
  s.addText("WE WILL NOT", {
    x: 7.6,
    y: 1.55,
    w: 4.95,
    h: 0.35,
    fontFace: B,
    fontSize: 13,
    color: C.gold,
    bold: true,
    charSpacing: 1.8,
    margin: 0,
  });
  const wont = [
    "Fabricate accuracy or hard-code predictions",
    "Call NASA-derived risk “ground-truth bug severity”",
    "Leak test data into scaling, SMOTE, or GA",
    "Report training scores as test performance",
    "Claim superiority before experiments finish",
    "Hide preprocessing decisions from the log",
  ];
  wont.forEach((line, i) => {
    s.addText(line, {
      x: 7.6,
      y: 2.1 + i * 0.65,
      w: 4.9,
      h: 0.55,
      fontFace: B,
      fontSize: 15,
      color: C.wh,
      margin: 0,
    });
  });
  footer(s, 12);

  // ─────────────────────────────────────────────
  // 13. THANK YOU
  // ─────────────────────────────────────────────
  s = pres.addSlide();
  s.background = { color: C.off };
  goldBar(s);
  sectionKicker(s, "CLOSE");
  title(s, "Thank you");

  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.8,
    y: 1.7,
    w: 9.7,
    h: 4.5,
    fill: { color: C.card },
    shadow: { type: "outer", color: "000000", blur: 8, offset: 2, angle: 135, opacity: 0.08 },
  });
  s.addShape(pres.shapes.RECTANGLE, {
    x: 1.8,
    y: 1.7,
    w: 9.7,
    h: 0.1,
    fill: { color: C.gold },
  });
  s.addText("Questions, discussion, and feedback are welcome.", {
    x: 2.15,
    y: 2.5,
    w: 9.0,
    h: 0.7,
    fontFace: H,
    fontSize: 26,
    color: C.navy,
    bold: true,
    align: "center",
    margin: 0,
  });
  s.addText(
    "NeuroDefect is an explainable decision-support prototype for software defect prediction.\nRisk severity is derived from neural probability and fuzzy inference — not NASA ground-truth bug severity.",
    {
      x: 2.3,
      y: 3.4,
      w: 8.7,
      h: 1.4,
      fontFace: B,
      fontSize: 16,
      color: C.tx,
      align: "center",
      margin: 0,
    }
  );
  s.addText("A Hybrid Neuro-Fuzzy-GA Decision Support System", {
    x: 2.15,
    y: 5.05,
    w: 9.0,
    h: 0.4,
    fontFace: B,
    fontSize: 14,
    color: C.teal,
    italic: true,
    align: "center",
    margin: 0,
  });
  footer(s, 13);

  await pres.writeFile({ fileName: "NeuroDefect_Project_Presentation.pptx" });
  console.log("Wrote NeuroDefect_Project_Presentation.pptx");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

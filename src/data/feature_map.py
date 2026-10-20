from __future__ import annotations

DISPLAY_NAMES: dict[str, str] = {
    "loc": "LOC_Total",
    "v(g)": "Cyclomatic_Complexity",
    "ev(g)": "Essential_Complexity",
    "iv(g)": "Design_Complexity",
    "n": "Halstead_Length",
    "v": "Halstead_Volume",
    "l": "Halstead_Level",
    "d": "Halstead_Difficulty",
    "i": "Halstead_Intelligence",
    "e": "Halstead_Effort",
    "b": "Error_Estimate",
    "t": "Halstead_Time",
    "lOCode": "Line_Count",
    "lOComment": "Comment_Lines",
    "lOBlank": "Blank_Lines",
    "locCodeAndComment": "Code_And_Comment",
    "uniq_Op": "Unique_Operators",
    "uniq_Opnd": "Unique_Operands",
    "total_Op": "Total_Operators",
    "total_Opnd": "Total_Operands",
    "branchCount": "Branch_Count",
}

FUZZY_ALIASES: dict[str, list[str]] = {
    "complexity": [
        "v(g)",
        "cyclomatic_complexity",
        "mccabe_complexity",
        "complexity",
        "vg",
        "wmc",
    ],
    "loc": ["loc", "loc_total", "locode", "loc_code", "nloc", "loc_blank"],
    "branch": ["branchcount", "branch_count", "branch"],
    "volume": ["v", "halstead_volume", "volume"],
    "difficulty": ["d", "halstead_difficulty", "difficulty"],
    "effort": ["e", "halstead_effort", "effort"],
}

TARGET_ALIASES = {
    "defects",
    "defect",
    "buggy",
    "bug",
    "label",
    "class",
    "defective",
    "problems",
    "is_defective",
    "isdefective",
    "faulty",
}

ID_ALIASES = {"id", "module_id", "module", "name", "filename", "file", "path"}


def display_name(column: str) -> str:
    return DISPLAY_NAMES.get(column, column)


def normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {normalize(col): col for col in columns}
    for alias in aliases:
        key = normalize(alias)
        if key in normalized:
            return normalized[key]
    return None


def map_fuzzy_inputs(columns: list[str]) -> dict[str, str]:
    """Map conceptual fuzzy inputs to actual dataset columns. Skip missing ones."""
    mapping: dict[str, str] = {}
    for concept, aliases in FUZZY_ALIASES.items():
        found = resolve_column(columns, aliases)
        if found is not None:
            mapping[concept] = found
    return mapping

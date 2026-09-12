from api.citations import cited_answer

SOURCES = ["Planes\nStarter cuesta 49 euros.", "Soporte\nAtención por email."]


def test_selects_only_declared_sources() -> None:
    answer, sources = cited_answer("Cuesta 49 euros.\nFUENTES: [1]", SOURCES)

    assert answer == "Cuesta 49 euros."
    assert sources == [SOURCES[0]]


def test_removes_duplicate_source_indices() -> None:
    _, sources = cited_answer("Respuesta.\nFUENTES: [2, 2, 1]", SOURCES)

    assert sources == [SOURCES[1], SOURCES[0]]


def test_empty_marker_returns_no_sources() -> None:
    answer, sources = cited_answer("No lo sé.\nFUENTES: []", SOURCES)

    assert answer == "No lo sé."
    assert sources == []


def test_missing_or_invalid_marker_falls_back_to_all_sources() -> None:
    assert cited_answer("Respuesta sin marcador", SOURCES)[1] == SOURCES
    assert cited_answer("Respuesta.\nFUENTES: [3]", SOURCES)[1] == SOURCES
    assert cited_answer("Respuesta.\nFUENTES: [uno]", SOURCES)[1] == SOURCES

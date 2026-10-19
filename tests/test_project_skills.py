"""Project skill discovery and critical workflow regression checks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_skills_have_portable_frontmatter_and_real_links() -> None:
    skills = sorted((ROOT / ".pi/skills").glob("*/SKILL.md"))
    assert len(skills) == 2
    for path in skills:
        text = path.read_text()
        assert text.startswith("---\nname: " + path.parent.name + "\n")
        assert "\ndescription: " in text
        assert len(text.splitlines()) < 150
    development = skills[0].read_text()
    assert "configs/innovation1-smoke.json" in development
    assert (ROOT / "configs/innovation1-smoke.json").is_file()
    workflow = skills[1].read_text()
    assert "Windows不commit/push" in workflow
    assert "runs/received/<run-id>/" in workflow


def test_result_and_backup_directories_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for folder in ("backups", "runs", "outputs", "results", "downloads", "validation"):
        assert f"/{folder}/" in ignored

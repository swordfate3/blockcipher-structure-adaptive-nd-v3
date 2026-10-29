"""Project skill discovery and critical workflow regression checks."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_project_skills_have_portable_frontmatter_and_real_links() -> None:
    skills = {p.parent.name: p for p in (ROOT / ".pi/skills").glob("*/SKILL.md")}
    assert set(skills) == {
        "cipher-experiment-development", "cipher-git-result-workflow", "cipher-research-workflow",
    }
    for path in skills.values():
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\nname: " + path.parent.name + "\n")
        assert "\ndescription: " in text
        assert len(text.splitlines()) < 150
    development = skills["cipher-experiment-development"].read_text(encoding="utf-8")
    assert "configs/innovation1-smoke.json" in development
    assert (ROOT / "configs/innovation1-smoke.json").is_file()
    workflow = skills["cipher-git-result-workflow"].read_text(encoding="utf-8")
    assert "Windows不commit/push" in workflow
    assert "runs/received/<run-id>/" in workflow
    assert "允许通过SSH执行已Git同步的CLI启动指令" in workflow
    assert "不通过SSH上传任务包" in workflow
    assert "授权后可经SSH调用远端已Git同步的CLI" in development


@pytest.mark.parametrize("relative", [
    "SKILL.md", "assets/research-cycle.md", "references/experiment-protocol.md",
])
def test_research_resources_have_resolvable_local_links(relative: str) -> None:
    path = ROOT / ".pi/skills/cipher-research-workflow" / relative
    assert path.is_file()
    for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" not in link and not link.startswith("#"):
            assert (path.parent / link.split("#")[0]).is_file(), (path, link)


@pytest.mark.parametrize(("resource", "required"), [
    ("SKILL.md", ("先做方法选择", "实质区别", "MLP可作基线", "不反复请求小步骤批准",
                  "默认正式重试0次", "真实key", "缺exit为UNKNOWN", "不自动续训")),
    ("assets/research-cycle.md", ("模板不是批准", "最多发布次数", "fit总数", "实际exit",
                                 "TEST哪些字段已见", "默认正式重试0", "权限终止")),
    ("references/experiment-protocol.md", ("当前没有该入口", "不能写出不存在的", "自动保存",
                                          "不是显著性", "git add -A", "未知消耗不能当0",
                                          "不视为授权", "取消优先")),
])
def test_research_workflow_retains_critical_boundaries(
    resource: str, required: tuple[str, ...],
) -> None:
    text = (ROOT / ".pi/skills/cipher-research-workflow" / resource).read_text(encoding="utf-8")
    for term in required:
        assert term in text, (resource, term)


def test_result_and_backup_directories_are_ignored() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for folder in ("backups", "runs", "outputs", "results", "downloads", "validation"):
        assert f"/{folder}/" in ignored

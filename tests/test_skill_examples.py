"""Tests for skill example templates and documentation accuracy."""
from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.skill_context import SkillContext
from src.tools.skill_manager import SkillManager
from src.config.schema import ToolsConfig, ToolHost
from src.tools.executor import ToolExecutor
from src.tools.registry import TOOLS

SKILLS_DIR = Path("data/skills")
TEMPLATE_FILES = sorted(SKILLS_DIR.glob("*.template"))
TEMPLATE_NAMES = [t.name.replace(".py.template", "") for t in TEMPLATE_FILES]


# ---------- Template file structure ----------


class TestTemplateFilesExist:
    def test_example_skill_template_exists(self):
        assert (SKILLS_DIR / "example_skill.py.template").exists()

    def test_system_info_template_exists(self):
        assert (SKILLS_DIR / "system_info.py.template").exists()

    def test_http_health_check_template_exists(self):
        assert (SKILLS_DIR / "http_health_check.py.template").exists()

    def test_at_least_three_templates(self):
        templates = list(SKILLS_DIR.glob("*.template"))
        assert len(templates) >= 3, f"Expected 3+ templates, found {len(templates)}"


class TestTemplateSyntax:
    """Verify all templates are syntactically valid Python."""

    @pytest.mark.parametrize("template", TEMPLATE_FILES, ids=TEMPLATE_NAMES)
    def test_template_parses(self, template: Path):
        code = template.read_text()
        ast.parse(code, filename=str(template))

    @pytest.mark.parametrize("template", TEMPLATE_FILES, ids=TEMPLATE_NAMES)
    def test_template_has_skill_definition(self, template: Path):
        code = template.read_text()
        tree = ast.parse(code)
        names = [
            node.targets[0].id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ]
        assert "SKILL_DEFINITION" in names, f"{template.name} missing SKILL_DEFINITION"

    @pytest.mark.parametrize("template", TEMPLATE_FILES, ids=TEMPLATE_NAMES)
    def test_template_has_async_execute(self, template: Path):
        code = template.read_text()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute":
                return
        pytest.fail(f"{template.name}: missing async execute() function")


# ---------- SKILL_DEFINITION validation ----------


class TestSkillDefinitionStructure:
    """Verify SKILL_DEFINITION dicts have required keys."""

    @pytest.mark.parametrize("template", TEMPLATE_FILES, ids=TEMPLATE_NAMES)
    def test_definition_has_required_keys(self, template: Path):
        mod = _load_template_module(template)
        defn = mod.SKILL_DEFINITION
        assert isinstance(defn, dict)
        for key in ("name", "description", "input_schema"):
            assert key in defn, f"{template.name}: SKILL_DEFINITION missing '{key}'"

    @pytest.mark.parametrize("template", TEMPLATE_FILES, ids=TEMPLATE_NAMES)
    def test_input_schema_is_valid(self, template: Path):
        mod = _load_template_module(template)
        schema = mod.SKILL_DEFINITION["input_schema"]
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert "properties" in schema

    @pytest.mark.parametrize("template", TEMPLATE_FILES, ids=TEMPLATE_NAMES)
    def test_definition_name_is_valid_skill_name(self, template: Path):
        """SKILL_DEFINITION name must be a valid skill name (lowercase, underscores)."""
        mod = _load_template_module(template)
        import re
        pattern = re.compile(r"^[a-z][a-z0-9_]{0,49}$")
        assert pattern.match(mod.SKILL_DEFINITION["name"])


# ---------- API documentation accuracy ----------


class TestTemplateApiDocs:
    """Verify the example_skill.py.template documents correct method names."""

    def test_documents_run_on_host(self):
        code = _read_template("example_skill")
        assert "run_on_host" in code
        assert "run_ssh" not in code, "Old API name 'run_ssh' should not appear"

    def test_documents_remember_recall(self):
        code = _read_template("example_skill")
        assert "remember(" in code
        assert "recall(" in code
        assert "get_memory" not in code, "Old API name 'get_memory' should not appear"
        assert "set_memory" not in code, "Old API name 'set_memory' should not appear"

    def test_documents_execute_tool(self):
        code = _read_template("example_skill")
        assert "execute_tool" in code

    def test_documents_search_knowledge(self):
        code = _read_template("example_skill")
        assert "search_knowledge" in code

    def test_documents_post_file(self):
        code = _read_template("example_skill")
        assert "post_file" in code

    def test_documents_post_message(self):
        code = _read_template("example_skill")
        assert "post_message" in code

    def test_documents_http_helpers(self):
        code = _read_template("example_skill")
        assert "http_get" in code
        assert "http_post" in code

    def test_documents_scheduling(self):
        code = _read_template("example_skill")
        assert "schedule_task" in code

    def test_no_nonexistent_methods(self):
        """Template should not reference methods that don't exist on SkillContext."""
        code = _read_template("example_skill")
        for bad_name in ("run_ssh", "write_file", "get_memory", "set_memory"):
            assert bad_name not in code, f"Template references non-existent method '{bad_name}'"


class TestRegistrySkillDocs:
    """Verify the create_skill tool description documents key SkillContext methods."""

    def _get_create_skill_description(self):
        for tool in TOOLS:
            if tool["name"] == "create_skill":
                return tool["description"]
        pytest.fail("create_skill tool not found in TOOLS")

    def test_documents_run_on_host(self):
        desc = self._get_create_skill_description()
        assert "run_on_host" in desc

    def test_documents_execute_tool(self):
        desc = self._get_create_skill_description()
        assert "execute_tool" in desc

    def test_documents_search_knowledge(self):
        desc = self._get_create_skill_description()
        assert "search_knowledge" in desc

    def test_documents_post_file(self):
        desc = self._get_create_skill_description()
        assert "post_file" in desc

    def test_documents_ingest_document(self):
        desc = self._get_create_skill_description()
        assert "ingest_document" in desc

    def test_documents_search_history(self):
        desc = self._get_create_skill_description()
        assert "search_history" in desc

    def test_documents_schedule_task(self):
        desc = self._get_create_skill_description()
        assert "schedule_task" in desc

    def test_references_template_files(self):
        desc = self._get_create_skill_description()
        assert "template" in desc.lower()


# ---------- Skill loading and execution ----------


@pytest.fixture
def skill_mgr(tmp_path: Path) -> SkillManager:
    config = ToolsConfig(hosts={"testhost": ToolHost(address="10.0.0.1")})
    executor = ToolExecutor(config)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    return SkillManager(
        str(skills_dir), executor, memory_path=str(tmp_path / "memory.json"),
    )


class TestExampleSkillExecution:
    """Load and execute the example skill templates."""

    def test_hello_world_loads(self, skill_mgr: SkillManager):
        code = _read_template("example_skill")
        result = skill_mgr.create_skill("hello_world", code)
        assert "successfully" in result
        assert skill_mgr.has_skill("hello_world")

    @pytest.mark.asyncio
    async def test_hello_world_default(self, skill_mgr: SkillManager):
        code = _read_template("example_skill")
        skill_mgr.create_skill("hello_world", code)
        result = await skill_mgr.execute("hello_world", {})
        assert "Hello, world!" in result

    @pytest.mark.asyncio
    async def test_hello_world_with_name(self, skill_mgr: SkillManager):
        code = _read_template("example_skill")
        skill_mgr.create_skill("hello_world", code)
        result = await skill_mgr.execute("hello_world", {"name": "Heimdall"})
        assert "Hello, Heimdall!" in result

    def test_system_info_loads(self, skill_mgr: SkillManager):
        code = _read_template("system_info")
        result = skill_mgr.create_skill("system_info", code)
        assert "successfully" in result

    @pytest.mark.asyncio
    async def test_system_info_no_host(self, skill_mgr: SkillManager):
        code = _read_template("system_info")
        skill_mgr.create_skill("system_info", code)
        result = await skill_mgr.execute("system_info", {})
        assert "specify a host" in result.lower() or "available" in result.lower()

    @pytest.mark.asyncio
    async def test_system_info_with_host(self, skill_mgr: SkillManager):
        code = _read_template("system_info")
        skill_mgr.create_skill("system_info", code)
        with patch("src.tools.executor.run_ssh_command", new_callable=AsyncMock) as mock_ssh:
            mock_ssh.return_value = (0, "=== Hostname ===\nmyhost\n=== Uptime ===\nup 5 days")
            result = await skill_mgr.execute("system_info", {"host": "testhost"})
            assert "myhost" in result or "Hostname" in result

    def test_http_health_check_loads(self, skill_mgr: SkillManager):
        code = _read_template("http_health_check")
        result = skill_mgr.create_skill("http_health_check", code)
        assert "successfully" in result

    @pytest.mark.asyncio
    async def test_http_health_check_no_url(self, skill_mgr: SkillManager):
        code = _read_template("http_health_check")
        skill_mgr.create_skill("http_health_check", code)
        result = await skill_mgr.execute("http_health_check", {})
        assert "provide a url" in result.lower()


# ---------- SkillContext API completeness ----------


class TestSkillContextApiCompleteness:
    """Verify that all public methods on SkillContext are documented."""

    def _get_public_methods(self):
        return [
            name for name in dir(SkillContext)
            if not name.startswith("_") and callable(getattr(SkillContext, name))
        ]

    def test_all_methods_in_template_docs(self):
        """Every public SkillContext method should be mentioned in the example template."""
        code = _read_template("example_skill")
        methods = self._get_public_methods()
        missing = [m for m in methods if m not in code]
        assert not missing, f"Template missing documentation for: {missing}"


# ---------- Helpers ----------


def _read_template(name: str) -> str:
    path = SKILLS_DIR / f"{name}.py.template"
    return path.read_text()


def _load_template_module(template: Path):
    """Import a template file as a module by copying to a .py file."""
    # importlib can't load .template files, so copy to a temp .py file
    tmp = Path(tempfile.mktemp(suffix=".py"))
    try:
        shutil.copy2(template, tmp)
        mod_name = f"_test_template_{template.name.replace('.', '_')}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        spec = importlib.util.spec_from_file_location(mod_name, tmp)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        tmp.unlink(missing_ok=True)

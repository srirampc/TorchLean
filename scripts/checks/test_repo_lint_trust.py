"""Regressions for source trust, import reachability and documentation synchronization."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dependency_audit
import repo_lint


class TrustBoundaryTests(unittest.TestCase):
    def declarations(self, source):
        masked = repo_lint._mask_lean_comments_and_strings(source)
        return repo_lint.AXIOM_DECL_RE.findall(masked)

    def test_visibility_attributes_and_unicode_names(self):
        source = """\
axiom plain : True
public axiom visible : True
private axiom hidden : True
@[simp] axiom attributed : True
@[simp]
public axiom αβ : True
protected axiom Nested.claim : True
"""
        self.assertEqual(
            self.declarations(source),
            ["plain", "visible", "hidden", "attributed", "αβ", "Nested.claim"],
        )

    def test_comments_strings_and_other_declarations_are_ignored(self):
        source = '''\
-- public axiom lineComment : True
/- @[simp] axiom blockComment : True /-
private axiom nestedComment : True -/ -/
def text := "public axiom inString : True"
def axiomLike := 1
theorem regular : True := True.intro
public /- visibility comment -/ axiom actual : True
'''
        self.assertEqual(self.declarations(source), ["actual"])

    def test_doc_example_sync_preserves_opening_summary(self):
        rendered = repo_lint._render_docstring(
            ["/-- A mathematical summary.", "", "Further explanation.", "-/"],
            ["#check Nat"],
        )
        self.assertEqual(rendered, [
            "/--", "A mathematical summary.", "", "Further explanation.", "",
            "Example:", "```lean", "#check Nat", "```", "-/",
        ])
        self.assertEqual(repo_lint._render_docstring(rendered, ["#check Nat"]), rendered)


class ImportCoverageTests(unittest.TestCase):
    def test_qualified_imports_reach_dependencies_but_body_examples_do_not(self):
        directives = [
            "import",
            "import all",
            "meta import",
            "meta import all",
            "public import",
            "public import all",
            "public meta import",
            "public meta import all",
        ]
        for directive in directives:
            with self.subTest(directive=directive), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sources = {
                    "NN.lean": "module\npublic import NN.AxisSemantics\n",
                    "NN/AxisSemantics.lean": (
                        "module\n"
                        f"{directive} /- modifier boundary -/ NN.AxisCoordinates\n"
                        "import NN.After\n"
                        "def exampleText := \"import NN.Orphan\"\n"
                        "import NN.Orphan\n"  # Body syntax is not part of the module header.
                    ),
                    "NN/AxisCoordinates.lean": "module\nimport all NN.Coordinates\n",
                    "NN/Coordinates.lean": "module\n",
                    "NN/After.lean": "module\n",
                    "NN/Orphan.lean": "module\n",
                }
                for name, source in sources.items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(source)
                findings = []
                with (patch.object(repo_lint, "REPO_ROOT", root),
                      patch.object(repo_lint, "_iter_lean_files",
                                   return_value=[root / name for name in sources]),
                      patch.object(repo_lint, "LEAN_TYPECHECK_ROOTS", {"NN"}),
                      patch.object(repo_lint, "LEAN_TYPECHECK_GLOB_PREFIXES", ())):
                    repo_lint._check_lean_target_coverage(findings)
                self.assertEqual([finding.path for finding in findings], [root / "NN/Orphan.lean"])

    def test_import_modifiers_and_offsets_are_preserved(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=newline):
                source = newline.join([
                    "module", "", "public meta import all NN.Qualified.Name'", "import NN.After", "",
                ])
                header = repo_lint._lean_import_header(source)
                matches = list(repo_lint.LEAN_IMPORT_RE.finditer(header))
                self.assertEqual([match.group("module") for match in matches],
                                 ["NN.Qualified.Name'", "NN.After"])
                self.assertEqual(matches[0].group("public"), "public ")
                self.assertEqual(repo_lint._line_col(source, matches[0].start()), (3, 1))

    def test_dependency_audit_records_all_imports_and_public_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "NN.lean"
            path.write_text(
                "module\n"
                "import all NN.Coordinates\n"
                "public meta import all NN.AxisCoordinates\n"
                "-- import all NN.Comment\n"
                'def text := "import all NN.String"\n'
            )
            edges, _, _ = dependency_audit.parse_file(root, path)
        self.assertEqual([(edge.dst, edge.public, edge.line) for edge in edges], [
            ("NN.Coordinates", False, 2),
            ("NN.AxisCoordinates", True, 3),
        ])


if __name__ == "__main__":
    unittest.main()

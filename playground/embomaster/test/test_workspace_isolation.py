from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from playground.embomaster.core.utils.workspace_isolation import prepare_workspace_codebase


class WorkspaceIsolationBootstrapTest(unittest.TestCase):
    def test_prepare_workspace_codebase_bootstraps_first_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            session_dir = root / "session"
            source_dir = root / "source"
            bootstrap_dir = root / "bootstrap"

            source_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "only_in_source.txt").write_text("source\n", encoding="utf-8")

            (bootstrap_dir / "policy").mkdir(parents=True, exist_ok=True)
            (bootstrap_dir / "policy" / "seed.txt").write_text("bootstrap\n", encoding="utf-8")

            info = prepare_workspace_codebase(
                session_dir=session_dir,
                workspace_id="exp_001-r1",
                source_codebase_dir=source_dir,
                bootstrap_codebase_dir=bootstrap_dir,
                use_copy_plan_cache=False,
            )

            self.assertEqual("bootstrap", info.source_type)
            self.assertTrue((info.path / "policy" / "seed.txt").exists())
            self.assertFalse((info.path / "only_in_source.txt").exists())
            self.assertEqual([], info.large_dirs)

    def test_parent_workspace_takes_priority_over_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            session_dir = root / "session"
            source_dir = root / "source"
            bootstrap_dir = root / "bootstrap"

            source_dir.mkdir(parents=True, exist_ok=True)
            (bootstrap_dir / "policy").mkdir(parents=True, exist_ok=True)
            (bootstrap_dir / "policy" / "from_bootstrap.txt").write_text(
                "bootstrap\n",
                encoding="utf-8",
            )

            first_round = prepare_workspace_codebase(
                session_dir=session_dir,
                workspace_id="exp_001-r1",
                source_codebase_dir=source_dir,
                bootstrap_codebase_dir=bootstrap_dir,
                use_copy_plan_cache=False,
            )
            (first_round.path / "policy" / "from_parent.txt").write_text("parent\n", encoding="utf-8")

            alt_bootstrap_dir = root / "bootstrap_alt"
            (alt_bootstrap_dir / "policy").mkdir(parents=True, exist_ok=True)
            (alt_bootstrap_dir / "policy" / "only_in_alt_bootstrap.txt").write_text(
                "alt\n",
                encoding="utf-8",
            )

            second_round = prepare_workspace_codebase(
                session_dir=session_dir,
                workspace_id="exp_001-r2",
                source_codebase_dir=source_dir,
                bootstrap_codebase_dir=alt_bootstrap_dir,
                parent_workspace_id="exp_001-r1",
                use_copy_plan_cache=False,
            )

            self.assertEqual("parent", second_round.source_type)
            self.assertTrue((second_round.path / "policy" / "from_parent.txt").exists())
            self.assertFalse((second_round.path / "policy" / "only_in_alt_bootstrap.txt").exists())


if __name__ == "__main__":
    unittest.main()

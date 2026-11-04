"""Standard-library checks for backend metadata without constructing models."""

import dataclasses
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from src.backends import BackendDescriptor, describe_backend, list_backends, resolve_backend
from src.backends.capabilities import _BACKENDS


class BackendContractTests(unittest.TestCase):
    """Check the version 1 metadata and selection boundary."""

    def test_discovery_order_and_metadata(self):
        names = ("local_ar", "hf_ar", "u2_reference", "u2_native")
        self.assertIsInstance(list_backends(), tuple)
        self.assertEqual(list_backends(), names)
        self.assertEqual(list_backends(), names)
        expected = (
            ("existing", frozenset({"causal_forward"})),
            ("existing", frozenset({"causal_forward", "autoregressive_generate"})),
            ("planned", frozenset()),
            ("planned", frozenset()),
        )
        for name, (status, capabilities) in zip(names, expected, strict=True):
            with self.subTest(name=name):
                descriptor = describe_backend(name)
                self.assertIsInstance(descriptor, BackendDescriptor)
                self.assertEqual(descriptor.name, name)
                self.assertEqual(descriptor.contract_version, 1)
                self.assertIs(type(descriptor.contract_version), int)
                self.assertEqual(descriptor.status, status)
                self.assertEqual(descriptor.capabilities, capabilities)
                self.assertIsInstance(descriptor.capabilities, frozenset)

    def test_descriptors_and_registry_are_immutable(self):
        for name in list_backends():
            descriptor = describe_backend(name)
            for field, value in (
                ("name", "changed"),
                ("contract_version", 2),
                ("capabilities", frozenset()),
                ("status", "planned"),
            ):
                with self.subTest(name=name, field=field):
                    with self.assertRaises(dataclasses.FrozenInstanceError):
                        setattr(descriptor, field, value)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                del descriptor.name
            with self.assertRaises(AttributeError):
                descriptor.capabilities.add("training")
        with self.assertRaises(TypeError):
            _BACKENDS["local_ar"] = describe_backend("hf_ar")
        with self.assertRaises(TypeError):
            del _BACKENDS["u2_native"]
        self.assertEqual(describe_backend("local_ar").capabilities, frozenset({"causal_forward"}))

    def test_descriptor_copies_caller_owned_capabilities(self):
        names = {"causal_forward"}
        descriptor = BackendDescriptor("custom", capabilities=names)
        names.add("autoregressive_generate")
        self.assertEqual(descriptor.capabilities, frozenset({"causal_forward"}))
        self.assertIsInstance(descriptor.capabilities, frozenset)
        self.assertIsInstance(hash(descriptor), int)
        self.assertEqual(BackendDescriptor("custom").capabilities, frozenset())

    def test_default_selection_matches_backbone_truthiness(self):
        self.assertEqual(resolve_backend().name, "local_ar")
        for backbone_id, expected in (
            (None, "local_ar"),
            ("", "local_ar"),
            ("example/model", "hf_ar"),
            (" ", "hf_ar"),
        ):
            with self.subTest(backbone_id=backbone_id):
                result = resolve_backend(llm_backbone_id=backbone_id)
                self.assertEqual(result, describe_backend(expected))

    def test_explicit_selection(self):
        for backbone_id in (None, ""):
            self.assertEqual(
                resolve_backend(backend="local_ar", llm_backbone_id=backbone_id).name,
                "local_ar",
            )
        self.assertEqual(
            resolve_backend(backend="hf_ar", llm_backbone_id="example/model").name, "hf_ar"
        )

    def test_legacy_presets_do_not_select_hf(self):
        # Exercise actual presets without importing the model or optional stacks.
        from src.config import ModelConfig

        for preset in (
            "prism-nano",
            "prism-micro",
            "prism-small",
            "prism-granite-2b",
            "prism-phi4-mini",
        ):
            with self.subTest(preset=preset):
                config = ModelConfig.from_preset(preset)
                self.assertTrue(config.hf_model_id)
                self.assertIsNone(config.llm_backbone_id)
                self.assertEqual(
                    resolve_backend(llm_backbone_id=config.llm_backbone_id).name, "local_ar"
                )
        config = ModelConfig.from_preset("prism-olmo3-7b")
        self.assertEqual(resolve_backend(llm_backbone_id=config.llm_backbone_id).name, "hf_ar")

    def test_hf_model_id_is_not_a_resolver_argument(self):
        with self.assertRaisesRegex(TypeError, "hf_model_id"):
            resolve_backend(hf_model_id="example/model")

    def test_conflicting_selectors(self):
        for backbone_id in (None, ""):
            with self.subTest(backbone_id=backbone_id):
                with self.assertRaisesRegex(ValueError, "hf_ar.*nonempty llm_backbone_id"):
                    resolve_backend(backend="hf_ar", llm_backbone_id=backbone_id)
        for backbone_id in ("example/model", " "):
            with self.assertRaisesRegex(ValueError, "local_ar.*conflicts.*llm_backbone_id"):
                resolve_backend(backend="local_ar", llm_backbone_id=backbone_id)

    def test_malformed_and_unknown_backend_selectors(self):
        for name in ("", " ", "HF_AR", "missing", 0, 1, True, False, b"local_ar", [], {}):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "[Bb]ackend"):
                    describe_backend(name)
                with self.assertRaisesRegex(ValueError, "[Bb]ackend"):
                    resolve_backend(backend=name)
        with self.assertRaisesRegex(ValueError, "backend name"):
            describe_backend(None)

    def test_malformed_backbone_ids(self):
        for backbone_id in (0, 1, True, False, 1.0, b"example/model", [], {}, object()):
            with self.subTest(backbone_id=backbone_id):
                for backend in (None, *list_backends()):
                    with self.assertRaisesRegex(ValueError, "llm_backbone_id.*string or None"):
                        resolve_backend(backend=backend, llm_backbone_id=backbone_id)

    def test_version_validation(self):
        self.assertEqual(resolve_backend(contract_version=1).contract_version, 1)
        for version in (True, False, 0, -1, 2, 1.0, "1", None, [], {}):
            with self.subTest(version=version):
                for backend in (None, "u2_reference", "u2_native"):
                    with self.assertRaisesRegex(ValueError, "contract_version.*integer 1"):
                        resolve_backend(backend=backend, contract_version=version)

    def test_supported_capabilities_and_iterables(self):
        for requested in (
            (),
            [],
            set(),
            frozenset(),
            ["causal_forward"],
            {"causal_forward"},
            ("causal_forward", "causal_forward"),
            iter(["causal_forward"]),
        ):
            with self.subTest(requested=requested):
                self.assertEqual(resolve_backend(required_capabilities=requested).name, "local_ar")
        self.assertEqual(
            resolve_backend(
                llm_backbone_id="example/model",
                required_capabilities=("causal_forward", "autoregressive_generate"),
            ).name,
            "hf_ar",
        )

    def test_local_generation_is_unsupported(self):
        with self.assertRaisesRegex(
            ValueError, "local_ar.*does not support.*autoregressive_generate"
        ):
            resolve_backend(required_capabilities=["autoregressive_generate"])

    def test_malformed_capability_inputs(self):
        malformed = (
            None,
            True,
            1,
            "causal_forward",
            b"",
            bytearray(),
            {},
            {"causal_forward": True},
            [None],
            [1],
            [True],
            [""],
            [[]],
            [{}],
        )
        for requested in malformed:
            with self.subTest(requested=requested):
                for backend in (None, "u2_reference", "u2_native"):
                    with self.assertRaisesRegex(ValueError, "required_capabilities"):
                        resolve_backend(backend=backend, required_capabilities=requested)

    def test_unknown_capabilities_precede_planned_rejection(self):
        for capability in (
            "unknown",
            "CAUSAL_FORWARD",
            "image",
            "structured_decoder",
            "diffusion",
            "training",
            "xpu",
        ):
            for backend in (None, "hf_ar", "u2_reference", "u2_native"):
                with self.subTest(backend=backend, capability=capability):
                    with self.assertRaisesRegex(ValueError, "Unknown required_capabilities"):
                        resolve_backend(backend=backend, required_capabilities=[capability])

    def test_planned_backends_never_fall_back(self):
        for backend in ("u2_reference", "u2_native"):
            for backbone_id in (None, "", "example/model"):
                for requested in ((), ("causal_forward",), ("autoregressive_generate",)):
                    with self.subTest(
                        backend=backend, backbone_id=backbone_id, requested=requested
                    ):
                        with self.assertRaisesRegex(
                            NotImplementedError, f"{backend}.*planned.*no fallback"
                        ):
                            resolve_backend(
                                backend=backend,
                                llm_backbone_id=backbone_id,
                                required_capabilities=requested,
                            )

    def test_import_and_metadata_calls_without_site_packages_or_side_effects(self):
        root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent("""\
            import json
            import os
            import sys

            assert sys.flags.no_site
            sys.path.insert(0, sys.argv[1])
            violations = []
            forbidden = {"torch", "transformers", "huggingface_hub", "requests",
                         "src.model", "src.config", "src.train"}

            def audit(event, args):
                blocked = False
                if event == "import":
                    blocked = any(args[0] == name or args[0].startswith(name + ".")
                                  for name in forbidden)
                elif event.startswith(("socket.", "urllib.")):
                    blocked = True
                elif event in {"subprocess.Popen", "os.system", "os.posix_spawn"}:
                    blocked = True
                elif event == "open":
                    path, mode, flags = args
                    writes = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                    # Import machinery may read Python source/bytecode only.
                    blocked = bool(flags & writes) or not (
                        isinstance(path, str) and path.endswith((".py", ".pyc"))
                    )
                if blocked:
                    violations.append(event)
                    raise AssertionError("Forbidden metadata side effect: " + event)

            sys.addaudithook(audit)
            from src.backends import describe_backend, list_backends, resolve_backend

            for name in list_backends():
                describe_backend(name)
            assert resolve_backend().name == "local_ar"
            assert resolve_backend(llm_backbone_id="not/a/real/model").name == "hf_ar"
            for name in ("u2_reference", "u2_native"):
                try:
                    resolve_backend(backend=name)
                except NotImplementedError:
                    pass
                else:
                    raise AssertionError("Planned backend resolved")
            assert not violations, violations
            assert not any(module == name or module.startswith(name + ".")
                           for module in sys.modules for name in forbidden)
            print(json.dumps(list_backends()))
        """)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, "-B", "-S", "-c", script, str(root)],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(list(Path(cwd).iterdir()), [])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), list(list_backends()))


if __name__ == "__main__":
    unittest.main()

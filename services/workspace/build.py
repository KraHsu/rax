"""One pipeline for every agent workspace: sources + recipe + setup -> files.

build() copies the recipe's template and skill layers, applies its role (documents,
variants, left-out skills and files), renders every markdown source against the
setup, and writes the tool contract. Callers add the session's extras (TASK.md,
demonstrations, CLI entry files) around it.
"""
import errno
import os
import shutil
import string
import time
from dataclasses import replace
from pathlib import Path

from .contract_files import write_contract
from .recipe import SKILL_SOURCE
from .templates import render

IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc')
# Templates may hold machine-local runtime/ state; never copy it.
TEMPLATE_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', 'runtime')


def remove_skill_tree(path):
    """Bounded retries for NFS metadata lag, without hiding permission errors."""
    for attempt in range(8):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            if not Path(path).exists():
                return
            if attempt == 7:
                raise
            time.sleep(0.05 * 2**attempt)
        except OSError as error:
            if error.errno not in (errno.ENOTEMPTY, errno.EBUSY) or attempt == 7:
                raise
            time.sleep(0.05 * 2**attempt)


def install_skills(skills, layers, source=SKILL_SOURCE, *, excluded=()):
    """Copy each layer's skill directories in order; later layers add or replace files."""
    skills = Path(skills)
    for layer in layers:
        root = Path(source) / layer
        if not root.is_dir():
            continue  # A layer may have no skills.
        for skill in sorted(path for path in root.iterdir() if path.is_dir()):
            if skill.name in excluded:
                continue
            if skill.is_symlink():
                raise ValueError('Skill sources must not be symlinks')
            shutil.copytree(skill, skills / skill.name, dirs_exist_ok=True, ignore=IGNORE)


def _no_link(path, what):
    if Path(path).is_symlink():
        raise ValueError(f'Workspace templates must not contain symlinked {what}')


def layers(recipe, setup):
    """The recipe's skill layers for this setup; a layer naming an unset axis is left out."""
    axes = setup.axes()
    return [layer.format(**axes) for layer in recipe.skill_layers
            if all(axes.get(name) is not None for _, name, _, _ in string.Formatter().parse(layer) if name)]


def apply_role(recipe, setup, workspace):
    """Documents and skills of this role, before rendering."""
    skills = workspace / '.agents/skills'
    omitted = {rule.skill for rule in recipe.omit_skills if rule.when(setup)}
    install_skills(skills, layers(recipe, setup), excluded=omitted)
    if recipe.agents_source != 'AGENTS.md':
        source = workspace / recipe.agents_source
        if not source.is_file() or source.is_symlink():
            raise ValueError(f'This workspace template has no {recipe.agents_source}')
        os.replace(source, workspace / 'AGENTS.md')
    for name in (*recipe.drop_documents, *recipe.role_documents):
        (workspace / name).unlink(missing_ok=True)
    for name in sorted(omitted):
        path = skills / name
        if path.exists():
            _no_link(path, 'skills')
            _no_link(skills, 'skills')
            remove_skill_tree(path)
    for kind in VARIANTS:
        for variant in sorted(skills.rglob(f'*.{kind}.md')):
            _no_link(variant, 'instructions')
            if kind == recipe.variant:
                os.replace(variant, variant.with_name(variant.name.removesuffix(f'.{kind}.md') + '.md'))
            else:
                variant.unlink()
    for relative in recipe.remove_files:
        path = skills / relative
        path.unlink(missing_ok=True)
        if path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()  # A references/ folder left empty.
    if setup.cli != 'codex':
        # agents/openai.yaml is Codex's skill metadata; other CLIs read only SKILL.md.
        for metadata in sorted(skills.glob('*/agents/openai.yaml')):
            _no_link(metadata, 'instructions')
            metadata.unlink()
            if not any(metadata.parent.iterdir()):
                metadata.parent.rmdir()


# Known skill-file variant kinds (<name>.<kind>.md).
VARIANTS = ('formal-episode',)


def render_sources(recipe, setup, workspace):
    """Render every markdown source in place; returns {relative path: block ids used}."""
    paths = list(workspace.glob('*.md'))
    for directory in recipe.render_dirs:
        if (workspace / directory).is_dir():
            paths.extend((workspace / directory).rglob('*.md'))
    blocks = {}
    for path in sorted(paths):
        _no_link(path, 'instructions')
        original = path.read_text()
        used = set()
        result = render(original, setup, vocab=recipe.vocab, used=used)
        if result != original:
            path.write_text(result)
        if used:
            blocks[path.relative_to(workspace).as_posix()] = sorted(used)
    return blocks


def manifest(recipe, setup, workspace, blocks):
    """Provenance of a built workspace (operator side, never in the workspace)."""
    import hashlib
    import json
    from .templates import BLOCKS
    files = {}
    for path in sorted(Path(workspace).rglob('*')):
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            relative = path.relative_to(workspace).as_posix()
            files[relative] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               **({'blocks': blocks[relative]} if relative in blocks else {})}
    return {'schema': 'workspace_build_v1', 'recipe': recipe.name, 'mode': 'auto-research',
            'options': sorted(setup.options), 'contract': setup.contract.manifest(),
            'blocks_registry_sha256': hashlib.sha256((BLOCKS / 'blocks.toml').read_bytes()).hexdigest(),
            'files': files}


def build(recipe, setup, workspace, *, copy_template=True):
    """Generate the workspace for `setup` into `workspace` (created if needed)."""
    workspace = Path(workspace)
    skills = workspace / '.agents/skills'
    if skills.is_symlink():
        raise ValueError('Workspace templates must not contain symlinked skills')
    if copy_template:
        shutil.copytree(recipe.template, workspace, dirs_exist_ok=True, ignore=TEMPLATE_IGNORE)
    setup = replace(setup, options=setup.options | recipe.options)
    apply_role(recipe, setup, workspace)
    blocks = render_sources(recipe, setup, workspace)
    write_contract(workspace, setup.contract, setup.options)
    return manifest(recipe, setup, workspace, blocks)

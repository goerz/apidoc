from __future__ import print_function

import importlib
import inspect
import os
import re
import sys
from functools import partial
from os import path

from docutils import nodes
from docutils.parsers.rst.states import RSTStateMachine, state_classes
from docutils.utils import Reporter as NullReporter
from docutils.utils import new_document
from jinja2 import FileSystemLoader
from jinja2.sandbox import SandboxedEnvironment
from sphinx.ext.autosummary import get_documenter
from sphinx.util.inspect import safe_getattr
from sphinx.util.osutil import walk

__author__ = "Michael Goerz (https://michaelgoerz.net)"

try:
    from sphinx.ext import apidoc  # Sphinx >= 1.7
    from sphinx.ext.apidoc import (
        makename,
        write_file,
        format_heading,
        format_directive,
        create_modules_toc_file,
        shall_skip,
        is_excluded,
    )
except ImportError:
    from sphinx import apidoc  # Sphinx < 1.7
    from sphinx.apidoc import (
        makename,
        write_file,
        format_heading,
        format_directive,
        create_modules_toc_file,
        shall_skip,
        is_excluded,
    )

periods_re = re.compile(r'\.(?:\s+)')

if False:
    # For type annotation
    from typing import Any, List, Tuple  # NOQA


def _warn(msg):
    print('WARNING: ' + msg, file=sys.stderr)


def _get_members(
        app,
        mod,
        typ=None,
        include_imported=False,
        out_format='names',
        in_list=None,
        known_refs=None,
):
    """Get (filtered) public/total members of the module or package `mod`.


    Returns:
        lists `public` and `items`. The lists contains the public and private +
        public members, as strings.
    """
    roles = {
        'function': 'func',
        'module': 'mod',
        'class': 'class',
        'exception': 'exc',
        'data': 'data',
    }
    # not included, because they cannot occur at modul level:
    #   'method': 'meth', 'attribute': 'attr', 'instanceattribute': 'attr'

    out_formats = ['names', 'fullnames', 'refs', 'table']
    if out_format not in out_formats:
        raise ValueError("out_format %s not in %r" % (out_format, out_formats))

    def check_typ(typ, mod, member):
        """Check if mod.member is of the desired typ"""
        if inspect.ismodule(member):
            return False
        try:  # Sphinx >= 1.7
            documenter = get_documenter(app=app, obj=member, parent=mod)
        except TypeError:  # Sphinx < 1.7
            documenter = get_documenter(obj=member, parent=mod)
        if typ is None:
            return True
        if typ == getattr(documenter, 'objtype', None):
            return True
        if hasattr(documenter, 'directivetype'):
            return roles[typ] == getattr(documenter, 'directivetype')
        return False

    def is_local(mod, member, name):
        """Check whether mod.member is defined locally in module mod"""
        if hasattr(member, '__module__'):
            return getattr(member, '__module__') == mod.__name__
        else:
            # we take missing __module__ to mean the member is a data object
            # it is recommended to filter data by e.g. __all__
            return True

    if typ is not None and typ not in roles:
        raise ValueError("typ must be None or one of %s" %
                         str(list(roles.keys())))
    items = []
    public = []
    if known_refs is None:
        known_refs = {}
    elif isinstance(known_refs, str):
        known_refs = getattr(mod, known_refs)
    if in_list is not None:
        try:
            in_list = getattr(mod, in_list)
        except AttributeError:
            in_list = []
    for name in dir(mod):
        if name.startswith('__'):
            continue
        try:
            member = safe_getattr(mod, name)
        except AttributeError:
            continue
        if check_typ(typ, mod, member):
            if in_list is not None:
                if name not in in_list:
                    continue
            if not (include_imported or is_local(mod, member, name)):
                continue
            if out_format in ['table', 'refs']:
                try:  # Sphinx >= 1.7
                    documenter = get_documenter(app=app,
                                                obj=member,
                                                parent=mod)
                except TypeError:  # Sphinx < 1.7
                    documenter = get_documenter(obj=member, parent=mod)
                role = roles.get(documenter.objtype, 'obj')
                ref = _get_member_ref_str(name,
                                          obj=member,
                                          role=role,
                                          known_refs=known_refs)
            if out_format == 'table':
                docsummary = extract_summary(member)
                items.append((ref, docsummary))
                if not name.startswith('_'):
                    public.append((ref, docsummary))
            elif out_format == 'refs':
                items.append(ref)
                if not name.startswith('_'):
                    public.append(ref)
            elif out_format == 'fullnames':
                fullname = _get_fullname(name, obj=member)
                items.append(fullname)
                if not name.startswith('_'):
                    public.append(fullname)
            else:
                assert out_format == 'names', str(out_format)
                items.append(name)
                if not name.startswith('_'):
                    public.append(name)
    if out_format == 'table':
        return _assemble_table(public), _assemble_table(items)
    else:
        return public, items


def _assemble_table(rows):
    if len(rows) == 0:
        return ''
    lines = []
    lines.append('.. list-table::')
    lines.append('')
    for row in rows:
        lines.append('   * - %s' % row[0])
        for col in row[1:]:
            lines.append('     - %s' % col)
    lines.append('')
    return lines


def extract_summary(obj):
    # type: (List[str], Any) -> str
    """Extract summary from docstring."""

    try:
        doc = inspect.getdoc(obj).split("\n")
    except AttributeError:
        doc = ''

    # Skip a blank lines at the top
    while doc and not doc[0].strip():
        doc.pop(0)

    # If there's a blank line, then we can assume the first sentence /
    # paragraph has ended, so anything after shouldn't be part of the
    # summary
    for i, piece in enumerate(doc):
        if not piece.strip():
            doc = doc[:i]
            break

    # Try to find the "first sentence", which may span multiple lines
    sentences = periods_re.split(" ".join(doc))  # type: ignore
    if len(sentences) == 1:
        summary = sentences[0].strip()
    else:
        summary = ''
        state_machine = RSTStateMachine(state_classes, 'Body')
        while sentences:
            summary += sentences.pop(0) + '.'
            node = new_document('')
            node.reporter = NullReporter('', 999, 4)
            node.settings.pep_references = None
            node.settings.rfc_references = None
            state_machine.run([summary], node)
            if not node.traverse(nodes.system_message):
                # considered as that splitting by period does not break inline
                # markups
                break

    return summary


def _get_member_ref_str(name, obj, role='obj', known_refs=None):
    """generate a ReST-formmated reference link to the given `obj` of type
    `role`, using `name` as the link text"""
    if known_refs is not None:
        if name in known_refs:
            return known_refs[name]
    ref = _get_fullname(name, obj)
    return ":%s:`%s <%s>`" % (role, name, ref)


def _get_fullname(name, obj):
    if hasattr(obj, '__qualname__'):
        try:
            ref = obj.__module__ + '.' + obj.__qualname__
        except AttributeError:
            ref = obj.__name__
        except TypeError:  # e.g. obj.__name__ is None
            ref = name
    elif hasattr(obj, '__name__'):
        try:
            ref = obj.__module__ + '.' + obj.__name__
        except AttributeError:
            ref = obj.__name__
        except TypeError:  # e.g. obj.__name__ is None
            ref = name
    else:
        ref = name
    return ref


def _get_mod_ns(app, name, fullname, includeprivate):
    """Return the template context of module identified by `fullname` as a
    dict"""
    ns = {  # template variables
        'name': name,
        'fullname': fullname,
        'members': [],
        'functions': [],
        'classes': [],
        'exceptions': [],
        'subpackages': [],
        'submodules': [],
        'doc': None,
    }
    p = 0
    if includeprivate:
        p = 1
    mod = importlib.import_module(fullname)
    ns['members'] = _get_members(app, mod)[p]
    ns['functions'] = _get_members(app, mod, typ='function')[p]
    ns['classes'] = _get_members(app, mod, typ='class')[p]
    ns['exceptions'] = _get_members(app, mod, typ='exception')[p]
    ns['data'] = _get_members(app, mod, typ='data')[p]
    ns['doc'] = mod.__doc__
    return ns


def add_get_members_to_template_env(app, template_env, fullname, opts):
    """Update the `template_env` with template variables"""

    def get_members(
            fullname,
            typ=None,
            include_imported=False,
            out_format='names',
            in_list=None,
            includeprivate=opts.includeprivate,
            known_refs=None,
    ):
        """Return a list of members

        Args:
            fullname (str): The full name of the module for which to get the
                members (including the dot-separated package path)
            typ (None or str): One of None, 'function', 'class', 'exception',
                'data'. If not None, only members of the corresponding type
                 will be returned
            include_imported (bool): If True, include members that are imported
                from other modules. If False, only return members that are
                defined directly in the module.
            out_format (str): One of 'names', 'fullnames', 'refs', and 'table'
            in_list (None or str): If not None, name of a module
                attribute (e.g. '__all__'). Only members whose names appears in
                the list will be returned.
            includeprivate (bool): If True, include members whose names starts
                with an underscore
            know_refs (None or dict or str): If not None, a mapping of names to
                rull rst-formatted references. If given as a str, the mapping
                will be taken from the module attribute of the given name. This
                is used only in conjunction with ``out_format=refs``, to
                override automatically detected reference location, or to
                provide references for object that cannot be located
                automatically (data objects).

        Returns:
            list: List of strings, depending on `out_format`.

            If 'names' (default), return a list of the simple names of all
            members.

            If 'fullnames', return a list of the fully qualified names
            of the members.

            If 'refs', return a list of rst-formatted links.

            If 'table', return a list of lines for a rst table similar to that
            generated by the autosummary plugin (left column is linked member
            names, right column is first sentence of the docstring)


        Note:
            For data members, it is not always possible to determine whther
            they are imported or defined locally. In this case, `in_list` and
            `known_refs` may be used to achieve the desired result.

            If using ``in_list='__all__'`` for a package you may also have to
            use ``include_imported=True`` to get the full list (as packages
            typically export members imported from their sub-modules)
        """
        mod = importlib.import_module(fullname)
        p = 0
        if includeprivate:
            p = 1
        members = _get_members(
            app,
            mod,
            typ=typ,
            include_imported=include_imported,
            out_format=out_format,
            in_list=in_list,
            known_refs=known_refs,
        )[p]
        return members

    template_env.globals['get_members'] = partial(get_members,
                                                  fullname=fullname)


def create_module_file(app, package, module, opts):
    # type: (str, str, Any) -> None
    """Generate RST for a top-level module (i.e., not part of a package)"""
    if not opts.noheadings:
        text = format_heading(1, '%s module' % module)
    else:
        text = ''
    # text += format_heading(2, ':mod:`%s` Module' % module)
    text += format_directive(module, package)

    template_loader = FileSystemLoader(opts.templates)
    template_env = SandboxedEnvironment(loader=template_loader)
    try:
        mod_ns = _get_mod_ns(
            app=app,
            name=module,
            fullname=module,
            includeprivate=opts.includeprivate,
        )
        template = template_env.get_template('module.rst')
        text = template.render(**mod_ns)
    except ImportError as e:
        _warn('failed to import %r: %s' % (module, e))
    add_get_members_to_template_env(app, template_env, module, opts)
    write_file(makename(package, module), text, opts)


def create_package_file(app, root, master_package, subroot, py_files, opts,
                        subs, is_namespace):
    # type: (str, str, str, List[str], Any, List[str], bool) -> None
    """Build the text of the file and write the file."""

    fullname = makename(master_package, subroot)

    template_loader = FileSystemLoader(opts.templates)
    template_env = SandboxedEnvironment(loader=template_loader)

    text = format_heading(
        1, ('%s package' if not is_namespace else "%s namespace") % fullname)

    if opts.modulefirst and not is_namespace:
        text += format_directive(subroot, master_package)
        text += '\n'

    # build a list of directories that are packages (contain an INITPY file)
    subs = [
        sub for sub in subs if path.isfile(path.join(root, sub, apidoc.INITPY))
    ]
    # if there are some package directories, add a TOC for theses subpackages
    if subs:
        text += format_heading(2, 'Subpackages')
        text += '.. toctree::\n\n'
        for sub in subs:
            text += '    %s.%s\n' % (makename(master_package, subroot), sub)
        text += '\n'

    submods = [
        path.splitext(sub)[0] for sub in py_files
        if not shall_skip(path.join(root, sub), opts) and sub != apidoc.INITPY
    ]

    try:
        package_ns = _get_mod_ns(
            app=app,
            name=subroot,
            fullname=fullname,
            includeprivate=opts.includeprivate,
        )
        package_ns['subpackages'] = subs
        package_ns['submodules'] = submods
    except ImportError as e:
        _warn('failed to import %r: %s' % (fullname, e))

    if submods:
        text += format_heading(2, 'Submodules')
        if opts.separatemodules:
            text += '.. toctree::\n\n'
            for submod in submods:
                modfile = makename(master_package, makename(subroot, submod))
                text += '   %s\n' % modfile

                # generate separate file for this module
                if not opts.noheadings:
                    filetext = format_heading(1, '%s module' % modfile)
                else:
                    filetext = ''
                filetext += format_directive(makename(subroot, submod),
                                             master_package)
                try:
                    mod_ns = _get_mod_ns(
                        app=app,
                        name=submod,
                        fullname=modfile,
                        includeprivate=opts.includeprivate,
                    )
                    template = template_env.get_template('module.rst')
                    add_get_members_to_template_env(app, template_env, modfile,
                                                    opts)
                    filetext = template.render(**mod_ns)
                except ImportError as e:
                    _warn('failed to import %r: %s' % (modfile, e))
                write_file(modfile, filetext, opts)
        else:
            for submod in submods:
                modfile = makename(master_package, makename(subroot, submod))
                if not opts.noheadings:
                    text += format_heading(2, '%s module' % modfile)
                text += format_directive(makename(subroot, submod),
                                         master_package)
                text += '\n'
        text += '\n'

    template = template_env.get_template('package.rst')
    add_get_members_to_template_env(app, template_env, fullname, opts)
    text = template.render(**package_ns)

    write_file(makename(master_package, subroot), text, opts)


def recurse_tree(app, rootpath, excludes, opts):
    # type: (str, List[str], Any) -> List[str]
    """Create rst file for all files in `rootpath`.

    Returns:
        List[str]: List of modules found in `rootpath`.
    """
    # check if the base directory is a package and get its name
    if apidoc.INITPY in os.listdir(rootpath):
        root_package = rootpath.split(path.sep)[-1]
    else:
        # otherwise, the base is a directory with packages
        root_package = None

    toplevels = []
    for root, subs, files in walk(rootpath, followlinks=opts.followlinks):
        # document only Python module files (that aren't excluded)
        py_files = sorted(f for f in files
                          if path.splitext(f)[1] in apidoc.PY_SUFFIXES
                          and not is_excluded(path.join(root, f), excludes))
        is_pkg = apidoc.INITPY in py_files
        is_namespace = (apidoc.INITPY not in py_files
                        and opts.implicit_namespaces)
        if is_pkg:
            py_files.remove(apidoc.INITPY)
            py_files.insert(0, apidoc.INITPY)
        elif root != rootpath:
            # only accept non-package at toplevel unless using implicit
            # namespaces
            if not opts.implicit_namespaces:
                del subs[:]
                continue
        # remove hidden ('.') and private ('_') directories, as well as
        # excluded dirs
        if opts.includeprivate:
            exclude_prefixes = ('.', )  # type: Tuple[str, ...]
        else:
            exclude_prefixes = ('.', '_')
        subs[:] = sorted(
            sub for sub in subs
            if (not sub.startswith(exclude_prefixes)
                and not is_excluded(path.join(root, sub), excludes)))

        if is_pkg or is_namespace:
            # we are in a package with something to document
            if (subs or len(py_files) > 1
                    or not shall_skip(path.join(root, apidoc.INITPY), opts)):
                subpackage = (root[len(rootpath):].lstrip(path.sep).replace(
                    path.sep, '.'))
                # if this is not a namespace or
                # a namespace and there is something there to document
                if not is_namespace or len(py_files) > 0:
                    # render a rst file for the package based on the
                    # package.rst template. If opts.separatemodules, also
                    # create on rst file per sub-module based on module.rst
                    # template
                    create_package_file(
                        app,
                        root,
                        root_package,
                        subpackage,
                        py_files,
                        opts,
                        subs,
                        is_namespace,
                    )
                    toplevels.append(makename(root_package, subpackage))
        else:
            # if we are at the root level, we don't require it to be a package
            assert root == rootpath and root_package is None
            # Generating the template variables requires that we can import
            # the module. If the module is not part of a package, the only way
            # to do this is to temporarily modify sys.path. This, and the
            # introduction of the `app` object is the only substantial change
            # of the recurse_tree function compared to Sphinx's original
            # implementation.
            sys.path.insert(0, rootpath)
            for py_file in py_files:
                if not shall_skip(path.join(rootpath, py_file), opts):
                    module = path.splitext(py_file)[0]
                    # render a rst file for the module according to the
                    # module.rst template
                    create_module_file(app, root_package, module, opts)
                    toplevels.append(module)
            sys.path.pop(0)  # undo temporary sys.path modification

    return toplevels


def normalize_excludes(rootpath, excludes):
    # type: (str, List[str]) -> List[str]
    """Normalize the excluded directory list."""
    return [path.abspath(exclude) for exclude in excludes]


class Options:
    """A dummy class for options, replacing the original CLI options."""

    def __init__(
            self,
            destdir,
            templates,
            suffix='rst',
            followlinks=False,
            includeprivate=False,
            implicit_namespaces=False,
            maxdepth=4,
            force=False,
            separatemodules=False,
            noheadings=False,
            modulefirst=False,
            toc_file=False,
    ):
        self.destdir = destdir
        self.templates = templates
        self.suffix = suffix
        self.dryrun = False
        self.followlinks = followlinks
        self.includeprivate = includeprivate
        self.implicit_namespaces = implicit_namespaces
        self.maxdepth = maxdepth
        self.force = force
        self.separatemodules = separatemodules
        self.noheadings = noheadings
        self.modulefirst = modulefirst
        self.toc_file = toc_file


def main(app, module_dir, output_dir, excludes, opts):
    """Generate API documentation with templates.

    Args:
        app: The Sphinx application object controlling the API generation
        module_dir (str): The absolute root path of the python packages and
            modules for which to generate the API
        output_dir (str): The  path of the folder to which to write the API rst
            files. If it does not exist, it will be created.
        excludes (list): List of files to exclude
        opts (Options): Namespace object with options, see :class:`Options`.
    """
    if not path.isdir(module_dir):
        print('%s is not a directory.' % module_dir, file=sys.stderr)
        sys.exit(1)
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    rootpath = path.abspath(module_dir)
    excludes = normalize_excludes(rootpath, excludes)
    modules = recurse_tree(app, rootpath, excludes, opts)
    if opts.toc_file:
        create_modules_toc_file(app, modules, opts)
    return 0

# binlearn Documentation

This directory contains the complete ReadTheDocs-compatible documentation for binlearn.

## Structure

```
docs/
├── Makefile                    # Build automation
├── requirements.txt           # Documentation dependencies
├── README.md                  # This file
└── source/
    ├── conf.py               # Sphinx configuration
    ├── index.rst             # Main documentation index
    ├── installation.rst      # Installation guide
    ├── quickstart.rst        # Quick start guide
    ├── contributing.rst      # Contribution guide
    ├── faq.rst              # Frequently asked questions
    ├── changelog.rst        # Version history
    ├── license.rst          # License information
    ├── troubleshooting.rst  # Troubleshooting guide
    ├── performance_tips.rst # Performance optimization
    ├── _static/
    │   └── custom.css       # Custom styling
    ├── tutorials/
    │   ├── index.rst        # Tutorial index
    │   └── beginner_tutorial.rst  # Beginner tutorial
    ├── user_guide/
    │   ├── index.rst        # User guide index
    │   └── overview.rst     # Package overview
    ├── api/
    │   ├── index.rst        # API reference index
    │   └── methods/
    │       ├── index.rst    # Methods index
    │       ├── equal_width_binning.rst
    │       ├── equal_frequency_binning.rst
    │       ├── kmeans_binning.rst
    │       ├── supervised_binning.rst
    │       └── singleton_binning.rst
    └── examples/
        ├── index.rst        # Examples index
        └── basic_examples.rst # Basic usage examples
```

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Or install with the main package:

```bash
pip install -e ".[docs]"
```

### Build Commands

From the `docs/` directory:

```bash
# Build HTML documentation
make html

# Clean build artifacts
make clean

# Build with warnings as errors (for CI)
make strict

# Serve documentation locally
make serve

# Live rebuild on changes
make livehtml

# Check for broken links
make linkcheck
```

### Local Development

For live documentation development:

```bash
cd docs/
make livehtml
```

This will:
- Build the documentation
- Start a local server at http://127.0.0.1:8000
- Automatically rebuild when files change

### Output

Built documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser.

## Documentation Features

### Comprehensive Coverage

- **Installation Guide**: Step-by-step installation instructions
- **Quick Start**: Get up and running in minutes
- **Tutorials**: Progressive learning from beginner to advanced
- **User Guide**: Detailed explanations of concepts and methods
- **API Reference**: Complete API documentation with examples
- **Examples**: Real-world use cases and code samples
- **FAQ**: Answers to common questions
- **Troubleshooting**: Solutions to common problems
- **Performance Tips**: Optimization strategies

### Modern Documentation Standards

- **Sphinx-based**: Industry standard documentation generator
- **ReadTheDocs Theme**: Professional, responsive design
- **Cross-references**: Automatic linking between sections
- **Code Highlighting**: Syntax highlighting for all code blocks
- **Auto-generated API**: Documentation from docstrings
- **Search**: Full-text search functionality
- **Mobile-friendly**: Responsive design for all devices

### Code Examples

All code examples are:
- **Tested**: Examples are verified to work
- **Complete**: Full working examples, not snippets
- **Practical**: Real-world use cases
- **Progressive**: From simple to advanced

## Contributing to Documentation

### Writing Guidelines

1. **Use clear, simple language**
2. **Include working code examples**
3. **Add cross-references where appropriate**
4. **Follow the existing structure and style**
5. **Test all code examples**

### reStructuredText Format

Documentation uses reStructuredText (.rst) format. Key syntax:

```rst
Title
=====

Section
-------

Subsection
~~~~~~~~~~

**Bold text**
*Italic text*
``Code text``

.. code-block:: python

   # Python code block
   import binlearn
   
.. note::
   This is a note admonition
   
.. warning::
   This is a warning admonition
```

### Adding New Content

1. **Create .rst files** in the appropriate directory
2. **Add to toctree** in the parent index.rst
3. **Test locally** with `make html`
4. **Check links** with `make linkcheck`

### API Documentation

API documentation is auto-generated from docstrings. To add documentation for a new class:

1. **Write comprehensive docstrings** in the source code
2. **Create .rst file** in `api/` directory
3. **Use autoclass directive**:

```rst
.. autoclass:: binlearn.NewClass
   :members:
   :inherited-members:
   :show-inheritance:
```

## ReadTheDocs Integration

This documentation is designed to work seamlessly with ReadTheDocs:

- **conf.py**: Configured for RTD hosting
- **requirements.txt**: All dependencies specified
- **Sphinx extensions**: RTD-compatible extensions
- **Theme**: Uses sphinx_rtd_theme
- **Build process**: Standard Sphinx build

### Configuration for ReadTheDocs

The documentation will automatically build on ReadTheDocs with:

- **Python version**: 3.11 (specified in .readthedocs.yaml if needed)
- **Requirements**: Installed from docs/requirements.txt
- **Build**: Standard Sphinx HTML build
- **Theme**: sphinx_rtd_theme

## Quality Assurance

### Build Validation

- **Warnings as errors**: Use `make strict` to catch issues
- **Link checking**: Use `make linkcheck` to verify links
- **Local testing**: Always test locally before committing

### Content Review

- **Accuracy**: Verify all code examples work
- **Completeness**: Ensure all features are documented  
- **Clarity**: Write for the target audience
- **Consistency**: Follow established patterns and style

## Maintenance

### Regular Tasks

- **Update examples**: Keep code examples current
- **Check links**: Verify external links still work
- **Update versions**: Keep version numbers current
- **Review accuracy**: Ensure content matches current codebase

### Version Updates

When releasing new versions:

1. **Update changelog.rst**: Add new version entry
2. **Update version**: In conf.py if needed
3. **Add new features**: Document new functionality
4. **Review examples**: Ensure compatibility

This documentation system provides a solid foundation for comprehensive, professional documentation that grows with the project.

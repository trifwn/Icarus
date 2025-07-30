"""Documentation System

This module provides searchable documentation with examples and cross-references
for all ICARUS features and capabilities.
"""

import json
import re
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple


class DocumentationType(Enum):
    """Types of documentation."""

    USER_GUIDE = "user_guide"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    API_DOC = "api_doc"
    TROUBLESHOOTING = "troubleshooting"
    FAQ = "faq"
    EXAMPLE = "example"


@dataclass
class CodeExample:
    """Code example with explanation."""

    title: str
    description: str
    code: str
    language: str = "python"
    output: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "description": self.description,
            "code": self.code,
            "language": self.language,
            "output": self.output,
            "notes": self.notes,
        }


@dataclass
class SearchableDoc:
    """Searchable documentation entry."""

    id: str
    title: str
    content: str
    doc_type: DocumentationType
    category: str
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    examples: List[CodeExample] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None
    difficulty: str = "beginner"  # beginner, intermediate, advanced

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "doc_type": self.doc_type.value,
            "category": self.category,
            "tags": self.tags,
            "keywords": self.keywords,
            "examples": [ex.to_dict() for ex in self.examples],
            "see_also": self.see_also,
            "last_updated": self.last_updated,
            "difficulty": self.difficulty,
        }


class DocumentationSystem:
    """Manages searchable documentation with examples and cross-references."""

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("cli/learning/data")
        self.documents: Dict[str, SearchableDoc] = {}
        self.search_index: Dict[str, Set[str]] = {}
        self.category_index: Dict[str, List[str]] = {}

        # Initialize built-in documentation
        self._initialize_documentation()
        self._build_search_index()

    def _initialize_documentation(self) -> None:
        """Initialize built-in documentation."""
        # User guide documentation
        self._create_user_guide_docs()

        # Reference documentation
        self._create_reference_docs()

        # Tutorial documentation
        self._create_tutorial_docs()

        # FAQ documentation
        self._create_faq_docs()

        # Example documentation
        self._create_example_docs()

    def _create_user_guide_docs(self) -> None:
        """Create user guide documentation."""
        # Getting Started Guide
        getting_started = SearchableDoc(
            id="getting_started",
            title="Getting Started with ICARUS CLI",
            content="""Welcome to ICARUS CLI - your comprehensive tool for aircraft design and analysis.

## What is ICARUS?

ICARUS is a powerful aerodynamics software suite that provides:
- 2D airfoil analysis using XFoil and other solvers
- 3D aircraft analysis with AVL and GenuVP
- Optimization and parameter studies
- Workflow automation for complex analyses
- Comprehensive visualization and reporting

## First Steps

1. **Launch ICARUS**: Run `icarus` from your terminal
2. **Take the Welcome Tour**: Follow the guided tutorial for new users
3. **Try Your First Analysis**: Start with a simple airfoil analysis
4. **Explore Features**: Use the navigation menu to discover capabilities
5. **Get Help**: Press Ctrl+H anytime for assistance

## Key Concepts

- **Analyses**: Individual computational studies (airfoil, aircraft, etc.)
- **Workflows**: Automated sequences of analyses
- **Results**: Data and visualizations from completed analyses
- **Projects**: Collections of related analyses and workflows

## Navigation Tips

- Use arrow keys or Tab to navigate menus
- Press Enter to select items
- Use number keys for quick menu selection
- Press Escape to go back or cancel operations
- F5 refreshes the current screen""",
            doc_type=DocumentationType.USER_GUIDE,
            category="Getting Started",
            tags=["introduction", "basics", "navigation"],
            keywords=["getting started", "first time", "introduction", "basics"],
            difficulty="beginner",
        )
        self.documents[getting_started.id] = getting_started

        # Interface Guide
        interface_guide = SearchableDoc(
            id="interface_guide",
            title="Interface Guide",
            content="""The ICARUS CLI interface is designed for efficiency and ease of use.

## Main Components

### Dashboard
Your central workspace showing:
- Recent analyses and results
- Quick access to common tasks
- System status and notifications
- Project overview

### Analysis Screen
Configure and run analyses:
- Select analysis type (airfoil, aircraft, etc.)
- Set parameters and operating conditions
- Choose solvers and methods
- Monitor progress and view results

### Results Screen
View and analyze results:
- Interactive plots and charts
- Data tables and summaries
- Export options for reports
- Comparison tools

### Workflow Screen
Create and manage workflows:
- Visual workflow builder
- Template library
- Execution monitoring
- Sharing and collaboration

### Settings Screen
Customize your experience:
- Theme and appearance
- Default parameters
- Solver configurations
- User preferences

## Keyboard Shortcuts

### Global Shortcuts
- Ctrl+H: Show help
- Ctrl+Q: Quit application
- F5: Refresh current screen
- F1: Context-sensitive help
- Ctrl+S: Save current work

### Navigation Shortcuts
- Arrow keys: Move between items
- Tab/Shift+Tab: Navigate forward/backward
- Enter: Select current item
- Escape: Go back or cancel
- 1-9: Quick menu selection

### Analysis Shortcuts
- Ctrl+R: Run analysis
- Ctrl+N: New analysis
- Ctrl+O: Open analysis
- Ctrl+E: Export results""",
            doc_type=DocumentationType.USER_GUIDE,
            category="Interface",
            tags=["interface", "navigation", "shortcuts"],
            keywords=["interface", "navigation", "keyboard", "shortcuts", "gui"],
            difficulty="beginner",
        )
        self.documents[interface_guide.id] = interface_guide

    def _create_reference_docs(self) -> None:
        """Create reference documentation."""
        # XFoil Reference
        xfoil_ref = SearchableDoc(
            id="xfoil_reference",
            title="XFoil Analysis Reference",
            content="""XFoil is a 2D airfoil analysis program that solves the viscous/inviscid interaction equations.

## Parameters

### Reynolds Number
- **Range**: 1e4 to 1e8 (typical: 1e5 to 1e7)
- **Effect**: Controls boundary layer behavior
- **Low Re**: Thicker boundary layer, earlier separation
- **High Re**: Thinner boundary layer, delayed separation

### Mach Number
- **Range**: 0.0 to 0.8 (XFoil validity limit)
- **Effect**: Compressibility effects
- **Below 0.3**: Incompressible flow assumption valid
- **Above 0.3**: Compressibility becomes important

### Angle of Attack
- **Range**: Typically -20° to +20°
- **Single Value**: For specific operating point
- **Range Analysis**: Sweep over multiple angles
- **Stall Limit**: Depends on airfoil and Reynolds number

### Transition
- **Free Transition**: Natural boundary layer transition
- **Fixed Transition**: Specify x/c location on upper/lower surfaces
- **Effect**: Influences drag and separation characteristics

## Output Data

### Force Coefficients
- **CL**: Lift coefficient
- **CD**: Drag coefficient
- **CM**: Moment coefficient (about quarter chord)

### Pressure Data
- **CP**: Pressure coefficient distribution
- **Upper/Lower**: Separate surface data
- **Stagnation Points**: Flow attachment/separation locations

### Boundary Layer Data
- **Displacement Thickness**: δ*
- **Momentum Thickness**: θ
- **Shape Factor**: H = δ*/θ
- **Skin Friction**: Cf distribution

## Convergence Tips

1. **Start Simple**: Begin with moderate angles and Reynolds numbers
2. **Check Geometry**: Ensure airfoil coordinates are clean
3. **Enable Viscous**: Use viscous analysis for realistic results
4. **Adjust Tolerance**: Modify convergence criteria if needed
5. **Panel Density**: Increase panels for complex geometries""",
            doc_type=DocumentationType.REFERENCE,
            category="Analysis",
            tags=["xfoil", "airfoil", "parameters", "reference"],
            keywords=["xfoil", "reynolds", "mach", "angle of attack", "convergence"],
            examples=[
                CodeExample(
                    title="Basic XFoil Analysis",
                    description="Set up a basic airfoil analysis with XFoil",
                    code="""# Configure XFoil analysis
analysis_config = {
    'airfoil': 'NACA 2412',
    'reynolds_number': 1e6,
    'mach_number': 0.1,
    'angle_range': {'start': -5, 'end': 15, 'step': 1},
    'solver': 'xfoil',
    'viscous': True
}

# Run analysis
results = run_airfoil_analysis(analysis_config)""",
                    language="python",
                    notes=[
                        "Reynolds number of 1e6 is typical for small aircraft",
                        "Mach 0.1 represents low-speed flight",
                        "Angle range covers typical operating envelope",
                    ],
                ),
            ],
            difficulty="intermediate",
        )
        self.documents[xfoil_ref.id] = xfoil_ref

    def _create_tutorial_docs(self) -> None:
        """Create tutorial documentation."""
        # First Analysis Tutorial
        first_analysis = SearchableDoc(
            id="first_analysis_tutorial",
            title="Your First Airfoil Analysis",
            content="""This tutorial walks you through your first airfoil analysis in ICARUS.

## Step 1: Select Analysis Type

1. Navigate to the Analysis screen
2. Choose "Airfoil Analysis" from the menu
3. Select "XFoil" as the solver

## Step 2: Choose an Airfoil

1. Click "Select Airfoil"
2. Choose "NACA 2412" from the database
3. Preview the airfoil geometry

## Step 3: Set Parameters

1. **Reynolds Number**: Enter 1000000 (1e6)
2. **Mach Number**: Enter 0.1
3. **Angle Range**: -5 to 15 degrees, step 1
4. **Analysis Type**: Viscous

## Step 4: Run Analysis

1. Click "Run Analysis"
2. Watch the progress indicator
3. Wait for completion (typically 30-60 seconds)

## Step 5: View Results

1. Examine the polar plot (CL vs Alpha)
2. Check the drag polar (CL vs CD)
3. Look at the L/D ratio plot
4. Note the maximum L/D angle

## Understanding Results

- **Stall Angle**: Where CL peaks and drops
- **Zero-Lift Angle**: Where CL = 0
- **Maximum L/D**: Most efficient operating point
- **Drag Minimum**: Lowest drag coefficient

## Next Steps

- Try different airfoils (NACA 0012, Clark Y)
- Vary Reynolds number to see effects
- Explore pressure distributions
- Learn about optimization studies""",
            doc_type=DocumentationType.TUTORIAL,
            category="Tutorials",
            tags=["tutorial", "airfoil", "first", "beginner"],
            keywords=["first analysis", "tutorial", "beginner", "step by step"],
            difficulty="beginner",
        )
        self.documents[first_analysis.id] = first_analysis

    def _create_faq_docs(self) -> None:
        """Create FAQ documentation."""
        faq = SearchableDoc(
            id="frequently_asked_questions",
            title="Frequently Asked Questions",
            content="""Common questions and answers about using ICARUS CLI.

## General Questions

### Q: What types of analyses can ICARUS perform?
A: ICARUS supports:
- 2D airfoil analysis (XFoil, panel methods)
- 3D aircraft analysis (AVL, GenuVP)
- Propulsion system analysis
- Mission performance analysis
- Multi-disciplinary optimization

### Q: Do I need to install external solvers?
A: Some solvers are included with ICARUS, others need separate installation:
- **Included**: Basic panel methods, optimization algorithms
- **External**: XFoil, AVL, GenuVP (installation guides provided)

### Q: Can I import my own airfoil geometries?
A: Yes, ICARUS supports multiple formats:
- Selig format (.dat files)
- Lednicer format
- UIUC database format
- Custom coordinate files

## Analysis Questions

### Q: Why does my XFoil analysis fail to converge?
A: Common causes and solutions:
- **High angle of attack**: Reduce maximum angle to 12-15°
- **Extreme Reynolds number**: Use values between 1e5 and 1e7
- **Poor geometry**: Check airfoil coordinates for issues
- **Wrong settings**: Enable viscous analysis

### Q: How do I interpret polar plots?
A: Key insights from polar plots:
- **CL vs Alpha**: Shows lift curve slope and stall behavior
- **CD vs Alpha**: Indicates drag rise and minimum drag
- **CL vs CD**: Drag polar shows efficiency at different lift levels
- **L/D vs Alpha**: Identifies most efficient operating angle

### Q: What Reynolds number should I use?
A: Depends on your application:
- **Model aircraft**: 1e5 to 5e5
- **General aviation**: 5e5 to 2e6
- **Commercial aircraft**: 1e6 to 1e7
- **High altitude**: Above 1e7

## Technical Questions

### Q: How accurate are the results?
A: Accuracy depends on several factors:
- **XFoil**: Very accurate for attached flow, less so near stall
- **Panel methods**: Good for inviscid flow, limited viscous effects
- **3D methods**: Depend on geometry complexity and mesh quality

### Q: Can I validate results against experimental data?
A: Yes, ICARUS provides tools for:
- Importing experimental data
- Plotting comparisons
- Statistical analysis of differences
- Uncertainty quantification

### Q: How do I cite ICARUS in publications?
A: Use the following citation format:
[Citation information would be provided here]

## Troubleshooting

### Q: The application won't start
A: Try these solutions:
1. Check Python installation and dependencies
2. Verify file permissions
3. Run from command line to see error messages
4. Check system requirements

### Q: Results look unrealistic
A: Verify your inputs:
1. Check airfoil geometry for errors
2. Ensure reasonable operating conditions
3. Validate solver settings
4. Compare with known data

### Q: How do I report bugs or request features?
A: Contact information and bug reporting procedures:
[Contact information would be provided here]""",
            doc_type=DocumentationType.FAQ,
            category="FAQ",
            tags=["faq", "questions", "troubleshooting"],
            keywords=["faq", "questions", "help", "problems", "issues"],
            difficulty="beginner",
        )
        self.documents[faq.id] = faq

    def _create_example_docs(self) -> None:
        """Create example documentation."""
        examples = SearchableDoc(
            id="analysis_examples",
            title="Analysis Examples",
            content="""Collection of practical analysis examples for common use cases.

## Example 1: Airfoil Comparison Study

Compare the performance of different airfoils for a specific application.

### Objective
Determine the best airfoil for a small UAV operating at Re = 200,000.

### Procedure
1. Select candidate airfoils (NACA 2412, Clark Y, Eppler 387)
2. Run analyses at identical conditions
3. Compare L/D ratios and stall characteristics
4. Consider manufacturing constraints

### Analysis Setup
- Reynolds Number: 200,000
- Mach Number: 0.05
- Angle Range: -2° to 12°
- Solver: XFoil with viscous analysis

## Example 2: Reynolds Number Study

Investigate how Reynolds number affects airfoil performance.

### Objective
Understand Re effects on NACA 0012 performance.

### Procedure
1. Fix airfoil geometry (NACA 0012)
2. Vary Reynolds number: 1e5, 2e5, 5e5, 1e6, 2e6
3. Compare polar curves
4. Analyze stall behavior changes

## Example 3: Optimization Study

Find optimal airfoil shape for specific requirements.

### Objective
Maximize L/D ratio at CL = 0.6 for Re = 1e6.

### Procedure
1. Define design variables (camber, thickness, etc.)
2. Set up optimization problem
3. Run optimization algorithm
4. Validate optimized design

## Example 4: Mission Analysis

Analyze aircraft performance for a specific mission profile.

### Objective
Determine fuel consumption for cross-country flight.

### Procedure
1. Define mission segments (takeoff, cruise, landing)
2. Set up aircraft configuration
3. Run mission analysis
4. Optimize for minimum fuel consumption""",
            doc_type=DocumentationType.EXAMPLE,
            category="Examples",
            tags=["examples", "case studies", "applications"],
            keywords=["examples", "case study", "comparison", "optimization"],
            examples=[
                CodeExample(
                    title="Airfoil Comparison Script",
                    description="Compare multiple airfoils at the same conditions",
                    code="""# Define airfoils to compare
airfoils = ['NACA 2412', 'Clark Y', 'Eppler 387']

# Common analysis parameters
params = {
    'reynolds_number': 200000,
    'mach_number': 0.05,
    'angle_range': {'start': -2, 'end': 12, 'step': 0.5},
    'solver': 'xfoil'
}

# Run comparison study
results = {}
for airfoil in airfoils:
    params['airfoil'] = airfoil
    results[airfoil] = run_airfoil_analysis(params)

# Compare maximum L/D ratios
for airfoil, result in results.items():
    max_ld = max(result['L_D_ratio'])
    print(f"{airfoil}: Max L/D = {max_ld:.2f}")""",
                    language="python",
                    output="""NACA 2412: Max L/D = 42.3
Clark Y: Max L/D = 38.7
Eppler 387: Max L/D = 51.2""",
                    notes=[
                        "Eppler 387 shows highest efficiency for this application",
                        "Consider stall characteristics and manufacturing constraints",
                        "Validate results with experimental data if available",
                    ],
                ),
            ],
            difficulty="intermediate",
        )
        self.documents[examples.id] = examples

    def _build_search_index(self) -> None:
        """Build search index for fast text search."""
        self.search_index.clear()
        self.category_index.clear()

        for doc_id, doc in self.documents.items():
            # Index by category
            if doc.category not in self.category_index:
                self.category_index[doc.category] = []
            self.category_index[doc.category].append(doc_id)

            # Index title words
            title_words = self._extract_words(doc.title)
            for word in title_words:
                if word not in self.search_index:
                    self.search_index[word] = set()
                self.search_index[word].add(doc_id)

            # Index content words (first 200 words to keep index manageable)
            content_words = self._extract_words(doc.content)[:200]
            for word in content_words:
                if len(word) > 2:  # Skip very short words
                    if word not in self.search_index:
                        self.search_index[word] = set()
                    self.search_index[word].add(doc_id)

            # Index tags and keywords
            for tag in doc.tags + doc.keywords:
                tag_words = self._extract_words(tag)
                for word in tag_words:
                    if word not in self.search_index:
                        self.search_index[word] = set()
                    self.search_index[word].add(doc_id)

    def _extract_words(self, text: str) -> List[str]:
        """Extract searchable words from text."""
        # Convert to lowercase and extract words
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return words

    def search(
        self,
        query: str,
        max_results: int = 10,
        category: str = None,
        doc_type: DocumentationType = None,
    ) -> List[Tuple[SearchableDoc, float]]:
        """Search documentation with relevance scoring."""
        query_words = self._extract_words(query)
        if not query_words:
            return []

        # Calculate relevance scores
        doc_scores: Dict[str, float] = {}

        for word in query_words:
            # Exact word matches
            if word in self.search_index:
                for doc_id in self.search_index[word]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 3.0

            # Partial word matches
            for index_word, doc_ids in self.search_index.items():
                if word in index_word or index_word in word:
                    similarity = len(word) / max(len(word), len(index_word))
                    for doc_id in doc_ids:
                        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + similarity

        # Filter by category and type
        filtered_docs = []
        for doc_id, score in doc_scores.items():
            if doc_id not in self.documents:
                continue

            doc = self.documents[doc_id]

            if category and doc.category.lower() != category.lower():
                continue

            if doc_type and doc.doc_type != doc_type:
                continue

            filtered_docs.append((doc, score))

        # Sort by relevance score
        filtered_docs.sort(key=lambda x: x[1], reverse=True)

        return filtered_docs[:max_results]

    def get_document(self, doc_id: str) -> Optional[SearchableDoc]:
        """Get a specific document."""
        return self.documents.get(doc_id)

    def get_documents_by_category(self, category: str) -> List[SearchableDoc]:
        """Get all documents in a category."""
        doc_ids = self.category_index.get(category, [])
        return [
            self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents
        ]

    def get_all_categories(self) -> List[str]:
        """Get all available categories."""
        return sorted(list(self.category_index.keys()))

    def get_related_documents(self, doc_id: str) -> List[SearchableDoc]:
        """Get documents related to the specified document."""
        if doc_id not in self.documents:
            return []

        doc = self.documents[doc_id]
        related = []

        # Get documents referenced in see_also
        for related_id in doc.see_also:
            if related_id in self.documents:
                related.append(self.documents[related_id])

        # Get documents with similar tags
        for other_id, other_doc in self.documents.items():
            if other_id == doc_id:
                continue

            # Check for common tags
            common_tags = set(doc.tags) & set(other_doc.tags)
            if len(common_tags) >= 2:  # At least 2 common tags
                related.append(other_doc)

        # Remove duplicates and limit results
        seen_ids = set()
        unique_related = []
        for related_doc in related:
            if related_doc.id not in seen_ids:
                unique_related.append(related_doc)
                seen_ids.add(related_doc.id)

        return unique_related[:5]  # Top 5 related documents

    def add_document(self, document: SearchableDoc) -> None:
        """Add a new document to the system."""
        self.documents[document.id] = document
        self._build_search_index()  # Rebuild index

    def update_document(self, doc_id: str, document: SearchableDoc) -> bool:
        """Update an existing document."""
        if doc_id not in self.documents:
            return False

        self.documents[doc_id] = document
        self._build_search_index()  # Rebuild index
        return True

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the system."""
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        self._build_search_index()  # Rebuild index
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get documentation system statistics."""
        doc_types = {}
        categories = {}
        difficulties = {}

        for doc in self.documents.values():
            # Count by type
            doc_type = doc.doc_type.value
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            # Count by category
            categories[doc.category] = categories.get(doc.category, 0) + 1

            # Count by difficulty
            difficulties[doc.difficulty] = difficulties.get(doc.difficulty, 0) + 1

        return {
            "total_documents": len(self.documents),
            "document_types": doc_types,
            "categories": categories,
            "difficulties": difficulties,
            "search_index_size": len(self.search_index),
        }

    def save_documentation(self, filepath: Path = None) -> None:
        """Save documentation to file."""
        if filepath is None:
            filepath = self.data_dir / "documentation.json"

        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": {
                doc_id: doc.to_dict() for doc_id, doc in self.documents.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_documentation(self, filepath: Path = None) -> None:
        """Load documentation from file."""
        if filepath is None:
            filepath = self.data_dir / "documentation.json"

        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)

            # Load documents
            self.documents.clear()
            for doc_id, doc_data in data.get("documents", {}).items():
                examples = []
                for ex_data in doc_data.get("examples", []):
                    example = CodeExample(
                        title=ex_data["title"],
                        description=ex_data["description"],
                        code=ex_data["code"],
                        language=ex_data.get("language", "python"),
                        output=ex_data.get("output"),
                        notes=ex_data.get("notes", []),
                    )
                    examples.append(example)

                document = SearchableDoc(
                    id=doc_data["id"],
                    title=doc_data["title"],
                    content=doc_data["content"],
                    doc_type=DocumentationType(doc_data["doc_type"]),
                    category=doc_data["category"],
                    tags=doc_data.get("tags", []),
                    keywords=doc_data.get("keywords", []),
                    examples=examples,
                    see_also=doc_data.get("see_also", []),
                    last_updated=doc_data.get("last_updated"),
                    difficulty=doc_data.get("difficulty", "beginner"),
                )
                self.documents[doc_id] = document

            self._build_search_index()

"""Build and execute the portable EDA notebook."""

import nbformat as nbf
from nbclient import NotebookClient
from src.data import ROOT


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Resume Screener AI — dataset audit\n\nReal Kaggle Resume Dataset. This notebook describes the raw data; vocabulary fitting and model selection occur only inside training folds. No raw personal resume text is displayed."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys, json\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nsys.path.insert(0, str(ROOT))\nimport pandas as pd\nfrom IPython.display import display, Image\nfrom src.eda import run\nstats = run()\ndf = pd.read_csv(ROOT / 'data/resumes.csv')\ndisplay(pd.Series(json.loads((ROOT / 'data/provenance.json').read_text())))"
        ),
        nbf.v4.new_markdown_cell(
            "## Category balance\nMacro F1 gives each category equal weight despite unequal support."
        ),
        nbf.v4.new_code_cell(
            "display(pd.Series(stats['category_counts'], name='resumes').to_frame())\ndisplay(Image(filename=str(ROOT / 'reports/figures/category_distribution.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Length and duplication\nWhitespace word counts include punctuation; preprocessing has its own token definition. Empty and duplicate cleaned resumes are excluded before the split."
        ),
        nbf.v4.new_code_cell(
            "display(pd.Series(stats['word_length_statistics'], name='words'))\nprint('Exact duplicate raw texts:', stats['raw_exact_duplicate_texts'])\ndisplay(Image(filename=str(ROOT / 'reports/figures/text_lengths.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Frequent cleaned terms\nThese are descriptive token counts, not the fitted model vocabulary. Common resume boilerplate and explicit job titles can make this benchmark easier than unseen real-world resumes."
        ),
        nbf.v4.new_code_cell(
            "display(pd.DataFrame(stats['frequent_terms_per_category']).fillna(0).astype(int))\ndisplay(Image(filename=str(ROOT / 'reports/figures/frequent_terms.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Interpretation\nThe source is a single website-derived English corpus. Category overlap, small minority classes, duplicates and title cues limit generalization. See results.json for deduplication counts and held-out evaluation; EDA is not evidence of deployment performance."
        ),
    ]
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbf.write(notebook, folder / "eda.ipynb")
    print("Executed notebooks/eda.ipynb")


if __name__ == "__main__":
    main()


import json

d = json.load(open("results/counts.json"))
c = d["classes"]
n_blocks = 3
h = -(-len(c) // n_blocks)
rows = []
for i in range(h):
    cells = []
    for b in range(n_blocks):
        j = i + b * h
        if j < len(c):
            cells.append(f"{c[j]} & {d['train'][j]} & {d['val'][j]} & {d['test'][j]}")
        else:
            cells.append(" & & & ")
    rows.append(" & ".join(cells) + " \\\\")

spec = "|".join(["lrrr"] * n_blocks)
head = " & ".join(["Breed & Tr & Va & Te"] * n_blocks) + " \\\\"
tot = f"Total: train {sum(d['train'])}, validation {sum(d['val'])}, test {sum(d['test'])}"

tex = ("\\begin{table}[H]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n"
       f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
       f"\\caption{{Images per breed in each split (Tr = training, Va = validation, Te = test). {tot}.}}\n"
       "\\label{tab:counts}\n\\end{table}\n")
open("report/counts_table.tex", "w").write(tex)
print("written report/counts_table.tex")
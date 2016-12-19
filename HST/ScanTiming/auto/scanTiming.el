(TeX-add-style-hook
 "scanTiming"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("scrartcl" "paper=letter" "fontsize=11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("nag" "l2tabu" "orthodox") ("babel" "english") ("siunitx" "use-xspace")))
   (TeX-run-style-hooks
    "latex2e"
    "nag"
    "scrartcl"
    "scrartcl10"
    "fouriernc"
    "babel"
    "amsmath"
    "amsfonts"
    "amsthm"
    "xspace"
    "graphicx"
    "sectsty"
    "booktabs"
    "siunitx"
    "fancyhdr")
   (TeX-add-symbols
    '("horrule" 1)
    "vs"
    "vr")
   (LaTeX-add-labels
    "tab:data"
    "tab:def"
    "eq:1"
    "eq:2"))
 :latex)


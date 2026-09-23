{ pkgs, inputs, ... }:

{
  # Offline figure vectoriser. Geometry comes from classical CV (OpenCV +
  # scikit-image), text from Tesseract and PP-FormulaNet, and the
  # render-and-compare refinement loop rasterises with resvg.
  # Run the CLI with:  devenv shell -- hybrid-vectorizer convert raster.png -o out.svg
  packages = with pkgs; [
    gnumake
    imagemagick
    tesseract
    inkscape
    resvg
    potrace
    fontconfig
    inputs.formulaocr-offline.packages.${pkgs.stdenv.hostPlatform.system}.default
  ];

  languages.python = {
    enable = true;
    package = pkgs.python313.withPackages (ps: with ps; [
      numpy
      scipy
      opencv4
      pillow
      scikit-image
      fonttools
    ]);
  };

  enterShell = ''
    export PYTHONPATH="$DEVENV_ROOT/src''${PYTHONPATH:+:$PYTHONPATH}"
    export PATH="$DEVENV_ROOT/bin:$PATH"
    python -c 'import cv2, numpy, scipy, skimage; print(f"hybrid-raster-vectorizer: opencv {cv2.__version__}  numpy {numpy.__version__}  skimage {skimage.__version__}")'
  '';
}

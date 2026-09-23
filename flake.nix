{
  description = "OCR-assisted semantic raster-to-SVG reconstruction";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    formulaocr-offline = {
      url = "github:dragonleopardpig/formulaocr-offline";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, formulaocr-offline, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      python = pkgs.python313.withPackages (ps: [
        ps.numpy
        ps.scipy
        ps.opencv4
        ps.pillow
        ps.scikit-image
        ps.fonttools
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          python
          pkgs.gnumake
          pkgs.imagemagick
          pkgs.tesseract
          pkgs.inkscape
          pkgs.resvg
          pkgs.potrace
          pkgs.fontconfig
          formulaocr-offline.packages.${system}.default
        ];

        shellHook = ''
          export PYTHONPATH="$PWD/src''${PYTHONPATH:+:$PYTHONPATH}"
        '';
      };
    };
}

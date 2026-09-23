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
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.gnumake
          pkgs.python313
          pkgs.imagemagick
          pkgs.tesseract
          pkgs.inkscape
          pkgs.potrace
          formulaocr-offline.packages.${system}.default
        ];
      };
    };
}

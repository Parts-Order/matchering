{
  description = "Matchering development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python3
            python3Packages.pip
            python3Packages.virtualenv
            libsndfile
            ffmpeg
          ];

          shellHook = ''
            echo "Matchering development environment"
            if [ ! -d "venv" ]; then
              python3 -m venv venv
            fi
            source venv/bin/activate
            pip install -r requirements.txt
            pip install -e .
          '';
        };
      });
}
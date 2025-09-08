{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            clippy
            just
            maturin
            rust-analyzer
            rustc
            uv
          ];

          shellHook = ''
            uv sync
            uv pip install -e .
            uv pip install -e ./hamming_core
            . .venv/bin/activate
          '';
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );

}

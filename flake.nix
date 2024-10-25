{
  inputs = {
    tgi-nix.url = "github:huggingface/text-generation-inference-nix";
    nixpkgs.follows = "tgi-nix/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      tgi-nix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          inherit (tgi-nix.lib) config;
          overlays = [
            tgi-nix.overlays.default
          ];
        };
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            buildInputs = [
              black
              pyright
              python3.pkgs.venvShellHook
            ] ++ (with cudaPackages; [
             cuda_nvcc
             cuda_nvtx
            ]);

            inputsFrom = [
              tgi-nix.packages.${system}.python3Packages.moe-kernels
            ];

            env = {
              CUDA_HOME = "${lib.getDev cudaPackages.cuda_nvcc}";
            };
            venvDir = "./.venv";
          };
      }
    );
}

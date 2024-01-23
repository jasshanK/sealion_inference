{
  description = "sealion inference with ggml dev";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  };

  outputs = { self, nixpkgs }: {
    devShell.x86_64-linux =
      let
        pkgs = nixpkgs.legacyPackages.x86_64-linux;
      in pkgs.mkShell {
        buildInputs = with pkgs; [
          ccls 
          pyright

          cmake

          python310
          python310Packages.pip
          python310Packages.virtualenv
          zlib
        ];

        LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:/run/opengl-driver/lib/:${pkgs.zlib}/lib";

        shellHook = ''
          alias create_venv="python -m venv .venv && mkdir .temp"
          alias load_venv="source .venv/bin/activate"
          alias pkg_setup="TMPDIR=./.temp/ pip3 install --require-virtualenv --cache-dir $TMPDIR -r requirements.txt"
        '';
      };
    };
  }

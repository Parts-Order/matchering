{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.virtualenv
    python3Packages.numpy
    python3Packages.scipy
    python3Packages.soundfile
    python3Packages.resampy
    python3Packages.statsmodels
    python3Packages.soxr
    python3Packages.matplotlib
    python3Packages.scikit-learn
    python3Packages.plotly
    python3Packages.kaleido
    libsndfile
    ffmpeg
    stdenv.cc.cc.lib
    glibc
  ];

  shellHook = ''
    echo "Setting up Matchering development environment..."

    # Set library paths for C++ stdlib
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.glibc}/lib:$LD_LIBRARY_PATH"

    # Use system packages instead of pip for numpy/scipy to avoid compilation issues
    export PYTHONPATH="${pkgs.python3Packages.numpy}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.scipy}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.soundfile}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.resampy}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.statsmodels}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.matplotlib}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.scikit-learn}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.plotly}/${pkgs.python3.sitePackages}:${pkgs.python3Packages.kaleido}/${pkgs.python3.sitePackages}:$PYTHONPATH"

    # Install the local matchering package in development mode
    pip install -e . --no-deps

    echo "Development environment ready!"
    echo "Run: python hyrax_sidechain_extract.py <audio_file>"
  '';
}

import sys
import os

computer_plateform = sys.platform
library_mapping = {
    "linux": r"../src/library/target/debug/liblibrary.so",
    "windows": r"../src/library/target/debug/liblibrary.dll",
    "darwin": r"../src/library/target/debug/liblibrary.dylib",
}

os.putenv("RUST_BACKTRACE", "full")

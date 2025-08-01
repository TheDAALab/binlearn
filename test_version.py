#!/usr/bin/env python3

import binning._version as v

print("🏷️  Version:", v.__version__)
print("📦 Version Tuple:", v.__version_tuple__)

info = v.get_version_info()
print("ℹ️  Version Info:")
for key, value in info.items():
    print(f"   {key}: {value}")

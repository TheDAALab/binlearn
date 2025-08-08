#!/usr/bin/env python3
"""
Final check for unused functions in remaining utils files.
"""


def check_function_usage(function_name):
    """Check if a function is used in the codebase."""
    import subprocess

    result = subprocess.run(
        [
            "grep",
            "-r",
            "--include=*.py",
            function_name,
            "/home/gykovacs/workspaces/binlearn/binlearn/",
        ],
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Filter out definition lines and __init__.py exports
    usage_lines = []
    for line in lines:
        if (
            line
            and "def " + function_name not in line
            and "__all__" not in line
            and "from ." not in line
        ):
            usage_lines.append(line)

    return len(usage_lines), usage_lines


# Check functions from data_handling.py
data_handling_functions = ["prepare_array", "return_like_input", "prepare_input_with_columns"]

# Check functions from parameter_conversion.py
parameter_conversion_functions = [
    "resolve_n_bins_parameter",
    "validate_numeric_parameter",
    "resolve_string_parameter",
    "validate_bin_number_parameter",
    "validate_bin_number_for_calculation",
]

print("USAGE CHECK FOR REMAINING UTILS FUNCTIONS:")
print("=" * 50)

all_used = True

for module_name, functions in [
    ("data_handling.py", data_handling_functions),
    ("parameter_conversion.py", parameter_conversion_functions),
]:
    print(f"\n📁 {module_name}:")

    for func in functions:
        count, usage = check_function_usage(func)

        # Consider used if more than just definition and exports
        if count > 2:  # definition + __init__ export + at least one real usage
            print(f"  ✅ {func}: {count} usages")
        else:
            print(f"  ❌ {func}: {count} usages (potentially unused)")
            print(f"     Usage: {usage}")
            all_used = False

print(f"\n{'✅ ALL FUNCTIONS ARE USED!' if all_used else '❌ Some functions may be unused'}")

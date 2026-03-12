#!/usr/bin/env python3
"""
Convert 4-dimensional optimization results to 6-dimensional format.
Extends each formula by appending two additional dimensions.
"""
import argparse
import json
import shutil
from pathlib import Path


def convert_formula_4d_to_6d(formula_4d, extension_strategy="repeat_last"):
    """
    Convert a 4D formula to 6D.
    
    Args:
        formula_4d: tuple of 4 floats (p0, p1, p2, p3) in prefix product form
        extension_strategy: how to extend to 6D
            - "repeat_last": (p0, p1, p2, p3, p3, p3)
            - "ones": (p0, p1, p2, p3, p3, p3) - maintains value
            - "decay": (p0, p1, p2, p3, p3*0.95, p3*0.9) - slight decay
    
    Returns:
        tuple of 6 floats
    """
    p0, p1, p2, p3 = formula_4d
    
    if extension_strategy == "repeat_last":
        # Keep the last value constant
        return (p0, p1, p2, p3, p3, p3)
    elif extension_strategy == "ones":
        # Same as repeat_last for prefix product form
        return (p0, p1, p2, p3, p3, p3)
    elif extension_strategy == "decay":
        # Slight decay for the extended dimensions
        p4 = p3 * 0.95
        p5 = p3 * 0.90
        return (p0, p1, p2, p3, p4, p5)
    else:
        raise ValueError(f"Unknown extension strategy: {extension_strategy}")


def convert_results(input_file, output_file, extension_strategy="repeat_last"):
    """
    Convert a 4D optimization results JSON to 6D format.
    
    Args:
        input_file: path to input JSON (4D)
        output_file: path to output JSON (6D)
        extension_strategy: strategy for extending dimensions
    """
    # Load 4D results
    with open(input_file, 'r') as f:
        results_4d = json.load(f)
    
    print(f"Loaded {len(results_4d)} results from {input_file}")
    
    # Convert each entry
    results_6d = []
    for entry in results_4d:
        formula_str = entry.get("formula")
        if not formula_str:
            print(f"Warning: Skipping entry without formula: {entry}")
            continue
        
        # Parse 4D formula
        try:
            if isinstance(formula_str, str):
                formula_4d = eval(formula_str)
            else:
                formula_4d = tuple(formula_str)
            
            if len(formula_4d) != 4:
                print(f"Warning: Formula is not 4D, skipping: {formula_str}")
                continue
            
            # Convert to 6D
            formula_6d = convert_formula_4d_to_6d(formula_4d, extension_strategy)
            
            # Create new entry
            new_entry = entry.copy()
            new_entry["formula"] = str(formula_6d)
            new_entry["original_4d_formula"] = formula_str
            
            results_6d.append(new_entry)
            
        except Exception as e:
            print(f"Error converting formula {formula_str}: {e}")
            continue
    
    # Save 6D results
    with open(output_file, 'w') as f:
        json.dump(results_6d, f, indent=4)
    
    print(f"Converted {len(results_6d)} results to 6D format")
    print(f"Saved to {output_file}")


def rename_with_dims_suffix(filepath, n_dims):
    """
    Rename a file to include dimension suffix before extension.
    e.g., optimization_results.json -> optimization_results_4dims.json
    
    Args:
        filepath: Path object or string
        n_dims: number of dimensions
    
    Returns:
        new Path object
    """
    filepath = Path(filepath)
    stem = filepath.stem
    suffix = filepath.suffix
    
    # Check if already has dims suffix
    if f"{n_dims}dims" in stem or f"_{n_dims}d" in stem:
        print(f"File {filepath.name} already has dimension suffix, skipping rename")
        return filepath
    
    new_name = f"{stem}_{n_dims}dims{suffix}"
    new_path = filepath.parent / new_name
    
    return new_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert 4D optimization results to 6D format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_file", type=str, 
                       help="Input JSON file with 4D results")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output JSON file for 6D results (default: auto-generated)")
    parser.add_argument("--extension_strategy", type=str, default="ones",
                       choices=["repeat_last", "ones", "decay"],
                       help="Strategy for extending to 6D")
    parser.add_argument("--rename_input", action="store_true",
                       help="Rename input file to include _4dims suffix")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup of input file before renaming")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        # Auto-generate: same directory, add _6dims suffix
        output_path = rename_with_dims_suffix(input_path, 6)
    
    # Create backup if requested
    if args.backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Convert 4D -> 6D
    convert_results(input_path, output_path, args.extension_strategy)
    
    # Rename input file if requested
    if args.rename_input:
        new_input_path = rename_with_dims_suffix(input_path, 4)
        if new_input_path != input_path:
            input_path.rename(new_input_path)
            print(f"Renamed input file: {input_path.name} -> {new_input_path.name}")
    
    print("\nConversion complete!")
    print(f"  4D results: {input_path if not args.rename_input else new_input_path}")
    print(f"  6D results: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

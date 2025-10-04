#!/bin/bash

# May 21 2024, RVP

# Converts all PNG files in a folder to PDF
# Usage: ./png2pdf.sh /path/to/folder

#input_folder="/media/Linux_1TB/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/system"
#input_folder="/Users/rahul/Dropbox (City College)/CUNY/Research/Bidisperse Project/analysis/figures/phim"
input_folder="/Users/rahul/City College Dropbox/Rahul Pandare/Sharepoint_Rahul_Surabh/png2pdf"

for file in "$input_folder"/*; do
    output_file="${file%.*}.pdf"
    sips -s format pdf "$file" --out "$output_file"
    #convert "$file" "$output_file"
    #rm "$file"
done

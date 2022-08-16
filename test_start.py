import sys
from openpyxl import load_workbook
wb = load_workbook('results_summary.xlsx')

# grab the active worksheet
ws = wb.active

# Add a blank row
ws.append(["", "", "", "", ""])

# Rows can also be appended
ws.append(["Name", "Lowest", "Mean", "Best Model", sys.argv[-1]])

# Save the file
wb.save("results_summary.xlsx")
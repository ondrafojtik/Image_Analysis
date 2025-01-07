# PowerShell script to run Python script
$pythonPath = "C:\Users\lucas\AppData\Local\Programs\Python\Python37\python.exe"
$scriptPath = ".\generate_model.py"

# Run the Python script
& $pythonPath $scriptPath

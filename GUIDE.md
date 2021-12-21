Ich weiss nicht, ob das ins Readme passt.

# Ordner
Benchmark_Results - Ordner mit Dateien bestehend aus den gemessenen Zeiten/ Verbrauch.
Converted_Models - Modelle, die in das ensprechende Format konvertiert wurden
Layer_Models - TF-Dateien und der Entsprechende Quellcode für diese. Our_Model kann möglicherweise als Super-Klasse verwendet werden

# Dateien
attempt.py beinhaltet meinen Quellcode um Zeiten zu messen. Es liest die Zahl aus conf aus, und führt so viele Iterationen der Inferenz  durch (bis jetzt Dummy Operation) (TO DO: Durchsatz).

In BenchmarkProblems will ich zufällige Eingaben (Mithilfe von Our_Model.getRandomInput()) erstellen und diese dann bei run() ans Netzwerk übergeben. Die Methode BenchMarkProblem.run() soll auch in attempt.py aufgerufen werden. (TO DO: Inferenz mehrmals aufrufen, oder mehr Daten direkt ans Netz übergeben?)

In Converter.py sollen Methoden landen, die Modelle in die entsprechende Darstellung überführen (TO DO: Funktionsweise bei  Coral rausfinden)

FirstInf.py enthält Methoden, mit denen ich die erste Inferenz auf dem Stick erzielt habe (Quelltext will ich nach BenchmarkProblems ziehen).

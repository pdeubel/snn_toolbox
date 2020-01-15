# Stand Konvertierung ANN zu SNN

## ANN

Das ANN ist ein vorhandenes Agentzoo Modell der Ant der Roboschool. Mit `ant-keras.py` kann dieses zu einem Keras
Modell exportiert werden, welches in die Toolbox eingesetzt wird.

## ANN parsen

Die Toolbox wird mit einer Konfigurationsdatei betrieben. In diesem Fall ist das `config-agentzoo.ini`. In `main.py`
wird die toolbox zusammen mit der Konfiguration gestartet. Das Parsen des Modells in ein internes (von der toolbox)
Modell funktioniert ohne Probleme, da die anschließende Überprüfung eine Genauigkeit von 100% ausgibt. Diese wird
mittels eines Datensatzes berechnet. Der Datensatz ist eine Reihe von zufälligen Eingaben in das originale ANN und
die Wahrheitswerte sind die resultierenden Ausgaben.

## SNN Simulation

Aus dem internen Modell wird dann ein SNN erstellt. Hier sind verschiedene Simulatoren möglich. Standard ist der 
INI Simulator der toolbox. Mit diesem wird eine Genauigkeit von >80% angegeben, mit Standardparametern. Inwiefern man
dieser Genauigkeit vertrauen kann ist nicht bekannt, da das enstandene SNN nicht mit der Roboschool getestet wurde.

Der andere Simulator, der eigentlich genutzt werden soll, ist `nest`. Bei diesem müssen die Parameter in den `cell`
Einstellungen geändert werden. `tau_refrac = 1` und `delay = 1` müssen gesetzt sein. (Diese müssen bei INI auf 0 stehen).
Außerdem muss `duration` in `simulation` größer als `1` sein. Dann läuft auch hier die Simulation durch, hat aber eine
Genauigkeit von 0%. Es müsste also weitere Einstellungen ausprobiert werden.

## Verschiedenes

Es ist wichtig nicht die aktuellste numpy Version zu installieren. Das erstellte Dockerfile funktioniert gut.

Außerdem sind verschiedene Funktionen angepasst, darunter in `pyNN_target_sim.py` in der Funktion `simulate()`:
`[np.linspace(0, self._duration, np.abs(int(self._duration * amplitude)))`. `np.abs(int())` ist hinzugefügt, wobei nur
`np.abs()` ausreicht. Es werden bei negativen Werten Fehler geworfen.

Ein nächster Schritt wäre eine aktuelle Version der Toolbox zu nehmen, ohne Veränderungen und darauf die Konfiguration,
das Keras Modell und den Testdatensatz anzuwenden und zu gucken ob es out-of-the-box funktioniert. Ansonsten Änderungen
im Sourcecode anzuwenden und versuchen Ausgaben des SNN bzw. des Simulator in Eingaben des OpenAi Gym umzuwandeln.
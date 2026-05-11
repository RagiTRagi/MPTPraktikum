Das Kalman-Filter
=================

Einführung in das Kalman-Filter
-------------------------------

Das Kalman-Filter ist ein rekursiver Schätzalgorithmus, der dazu verwendet wird, den Zustand eines dynamischen Systems aus einer Reihe unvollständiger und verrauschter Messdaten zu schätzen. Entwickelt wurde es in den 1960er Jahren von Rudolf E. Kálmán und hat seitdem breite Anwendung in verschiedenen technischen Disziplinen gefunden – von der Signalverarbeitung über die Robotik bis hin zur Navigation.

Im Kern kombiniert das Kalman-Filter ein mathematisches Modell des Systems mit tatsächlichen Messungen, um eine optimierte Schätzung des Systemzustands zu erzeugen. Es basiert auf zwei grundlegenden Phasen: der Vorhersage (Prädiktion) und der Korrektur (Update). In der Vorhersagephase wird der zukünftige Zustand des Systems auf Basis des Modells abgeschätzt. In der Korrekturphase wird diese Vorhersage mithilfe neuer Messdaten aktualisiert, wobei das Vertrauen in Modell und Messung durch Kovarianzmatrizen gesteuert wird.

Dank seiner Fähigkeit, auch bei starkem Messrauschen robuste Schätzungen zu liefern, ist das Kalman-Filter besonders in Anwendungen mit Echtzeitanforderungen und begrenzter Sensorpräzision unverzichtbar geworden.

Annahmen des Kalman-Filters
---------------------------

Damit das Kalman-Filter optimale Ergebnisse liefert, beruht es auf einer Reihe fundamentaler Annahmen über das zugrundeliegende System und die Messungen:

1. **Lineare Modelle**  
   Sowohl das Zustandsmodell als auch das Messmodell müssen linear sein. Das bedeutet:
   
   - Der Systemzustand entwickelt sich gemäß einer linearen Gleichung weiter:
     
     .. math::

        x_k = A x_{k-1} + B u_{k-1} + w_{k-1}
   
   - Die Messung hängt ebenfalls linear vom Zustand ab:
     
     .. math::

        z_k = H x_k + v_k

   Dabei sind :math:`A`, :math:`B` und :math:`H` bekannte Matrizen, :math:`w_k` das Prozessrauschen und :math:`v_k` das Messrauschen.

2. **Normalverteilte Zufallsgrößen**  
   Sowohl das Prozessrauschen :math:`w_k` als auch das Messrauschen :math:`v_k` werden als **weißes Rauschen mit Normalverteilung** angenommen:
   
   .. math::

      w_k \sim \mathcal{N}(0, Q), \quad v_k \sim \mathcal{N}(0, R)

   Diese Annahme stellt sicher, dass auch die geschätzten Zustände normalverteilt sind und das Filter eine geschlossene Form für Mittelwert und Kovarianz liefern kann.

3. **Unkorrelierte Rauschquellen**  
   Das Prozess- und das Messrauschen sind voneinander **unkorreliert**:
   
   .. math::

      E[w_k v_j^T] = 0 \quad \text{für alle } k, j

   Diese Unabhängigkeit ist entscheidend dafür, dass sich der Fehlerfortpflanzung sauber voneinander trennen lässt.

4. **Vollständige Kenntnis der Systemparameter**  
   Die Modellmatrizen (:math:`A`, :math:`B`, :math:`H`), sowie die Kovarianzmatrizen des Rauschens (:math:`Q`, :math:`R`) sind bekannt und konstant oder zumindest zeitabhängig aber vorab gegeben.

Diese Voraussetzungen sind in vielen realen Anwendungen nur näherungsweise erfüllt. Dennoch liefert das Kalman-Filter auch bei leicht verletzten Annahmen oft brauchbare Resultate. Für nichtlineare oder nicht-Gaussverteilungen existieren Erweiterungen wie das Extended Kalman Filter (EKF) oder Unscented Kalman Filter (UKF).

Filtergleichungen: Prädiktion und Update
----------------------------------------

.. image:: filtering.png
   :width: 800px
   :align: center
   :alt: Kalman-Filter Phasen

Das Kalman-Filter operiert in zwei Hauptphasen: **Prädiktion (Vorhersage)** und **Update (Korrektur)**. In jeder Zeitschrittiteration werden nacheinander diese beiden Schritte ausgeführt, um den Systemzustand möglichst genau zu schätzen.

**Prädiktionsschritt**

In der Prädiktionsphase wird der nächste Zustand des Systems basierend auf dem aktuellen Schätzwert und dem Modell vorhergesagt.

**Zustandsvorhersage:**

.. math::

   \hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_{k-1}

**Fehlerkovarianz-Vorhersage:**

.. math::

   P_{k|k-1} = A P_{k-1|k-1} A^T + Q

Hierbei bezeichnet:

- :math:`\hat{x}_{k|k-1}` den vorhergesagten Zustand zum Zeitpunkt :math:`k` basierend auf Information bis :math:`k-1`
- :math:`P_{k|k-1}` die vorhergesagte Kovarianzmatrix
- :math:`Q` die Kovarianzmatrix des Prozessrauschens

**Updateschritt (Korrektur)**

Sobald eine neue Messung :math:`z_k` eintrifft, wird die Vorhersage mit dieser Information korrigiert.

**Innovation (Messresiduum):**

.. math::

   y_k = z_k - H \hat{x}_{k|k-1}

**Innovationskovarianz:**

.. math::

   S_k = H P_{k|k-1} H^T + R

**Kalman-Gewinn:**

.. math::

   K_k = P_{k|k-1} H^T S_k^{-1}

**Zustandsaktualisierung:**

.. math::

   \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k y_k

**Kovarianzaktualisierung:**

.. math::

   P_{k|k} = (I - K_k H) P_{k|k-1}

Diese rekursive Struktur macht das Kalman-Filter besonders effizient, da es keine historischen Messdaten speichern muss. Stattdessen wird der Zustandsschätzer und die Kovarianzmatrix bei jedem neuen Schritt nur anhand der neuesten Informationen aktualisiert.

Initialisierung des Kalman-Filters
----------------------------------

Bevor das Kalman-Filter mit der rekursiven Schätzung beginnen kann, müssen der anfängliche Zustand und seine Unsicherheit spezifiziert werden. Diese **Initialisierung** spielt eine wichtige Rolle für die Konvergenzgeschwindigkeit und Genauigkeit des Filters in den ersten Zeitschritten.

**Zustandsschätzung aus der ersten Messung**

In vielen praktischen Anwendungen liegen keine exakten Informationen über den Anfangszustand vor. Eine gängige Methode besteht daher darin, den **ersten Messwert** :math:`z_0` als Startpunkt für die Zustandsschätzung zu verwenden. Dazu wird angenommen, dass der Messwert bereits eine (ggf. verrauschte) Beobachtung des tatsächlichen Zustands ist:

.. math::

   \hat{x}_{0|0} = H^{-1} z_0

Falls :math:`H` nicht invertierbar ist (z. B. bei nicht-vollständiger Beobachtung), kann alternativ eine Annäherung oder Projektion auf den beobachtbaren Raum genutzt werden. In vielen Fällen wird auch direkt gesetzt:

.. math::

   \hat{x}_{0|0} = z_0

sofern das Messmodell :math:`H = I` (Identitätsmatrix) ist, also der Messwert direkt dem Zustand entspricht.

**Initiale Kovarianzmatrix**

Die anfängliche Unsicherheit über den Zustand wird durch die Kovarianzmatrix :math:`P_{0|0}` beschrieben. Diese sollte so gewählt werden, dass sie die Unsicherheit der Startschätzung realistisch widerspiegelt:

- Ist der Anfangszustand **präzise bekannt**, kann :math:`P_{0|0}` klein gewählt werden (z. B. :math:`P_{0|0} = 0`).
- Ist der Anfangszustand **unsicher**, sollte :math:`P_{0|0}` große Werte enthalten (z. B. eine skalierte Einheitsmatrix mit großem Faktor).

Ein typischer Initialwert ist:

.. math::

   P_{0|0} = \sigma^2 I

wobei :math:`\sigma^2` eine heuristisch gewählte Anfangsvarianz ist.

**Bemerkung zur Konvergenz**

Auch wenn die Initialisierung ungenau ist, konvergiert das Kalman-Filter bei hinreichend informativen Messdaten in der Regel nach einigen Schritten gegen die optimale Schätzung. Dennoch kann eine sinnvolle Initialisierung das Verhalten in der Anfangsphase stark verbessern.



**Beispiel**: 1-dimensionales Kalman-Filter
-------------------------------------------

Dieses Beispiel zeigt die Funktionsweise des Kalman-Filters in einem einfachen 1D-Szenario, in dem der Zustand die Position eines sich gleichförmig bewegenden Objekts beschreibt. Wir nehmen an:

- Der Zustand ist die Position :math:`x`.
- Es gibt keine Kontrolleingabe :math:`u`.
- Das Systemmodell lautet: :math:`x_k = x_{k-1} + w_k` mit :math:`w_k \sim \mathcal{N}(0, Q)`
- Das Messmodell lautet: :math:`z_k = x_k + v_k` mit :math:`v_k \sim \mathcal{N}(0, R)`

**Gegeben:**

- Anfangsschätzung: :math:`\hat{x}_{0|0} = 0`
- Anfangskovarianz: :math:`P_{0|0} = 1.0`
- Prozessrauschen: :math:`Q = 1.0`
- Messrauschen: :math:`R = 2.0`
- Messwert bei :math:`k=1`: :math:`z_1 = 1.2`

---

**Schritt 1: Prädiktion**

.. math::

   \hat{x}_{1|0} = \hat{x}_{0|0} = 0.0

.. math::

   P_{1|0} = P_{0|0} + Q = 1.0 + 1.0 = 2.0

---

**Schritt 2: Update**

**Innovation:**

.. math::

   y_1 = z_1 - \hat{x}_{1|0} = 1.2 - 0.0 = 1.2

**Innovationskovarianz:**

.. math::

   S_1 = P_{1|0} + R = 2.0 + 2.0 = 4.0

**Kalman-Gewinn:**

.. math::

   K_1 = \frac{P_{1|0}}{S_1} = \frac{2.0}{4.0} = 0.5

**Zustandsaktualisierung:**

.. math::

   \hat{x}_{1|1} = \hat{x}_{1|0} + K_1 \cdot y_1 = 0.0 + 0.5 \cdot 1.2 = 0.6

**Kovarianzaktualisierung:**

.. math::

   P_{1|1} = (1 - K_1) \cdot P_{1|0} = (1 - 0.5) \cdot 2.0 = 1.0

**Ergebnis nach Schritt 1:**

- Geschätzte Position: :math:`\hat{x}_{1|1} = 0.6`
- Unsicherheit: :math:`P_{1|1} = 1.0`

**Anmerkung:**  
Dieses einfache Beispiel zeigt, wie das Kalman-Filter Messwerte mit Modellvorhersagen kombiniert. Die Schätzung liegt zwischen dem Modell (0.0) und der Messung (1.2), gewichtet durch das Vertrauen in beide Quellen. Da Mess- und Modellunsicherheit gleich groß sind, wird der Mittelwert gewählt.

Weitere Iterationen würden nach dem gleichen Schema fortfahren.


Mehrere Messungen zum selben Zeitpunkt
--------------------------------------

In vielen praktischen Anwendungen stehen mehrere Sensoren zur Verfügung, die **zeitgleich** Informationen über denselben Systemzustand liefern. Typische Beispiele sind GPS-, Lidar- und Radarsensoren in autonomen Fahrzeugen. Das Kalman-Filter lässt sich auf diese Situation elegant erweitern.

**Grundidee**

Wenn mehrere Messungen :math:`z_k^{(1)}, z_k^{(2)}, \dots, z_k^{(n)}` zum gleichen Zeitpunkt vorliegen, können diese entweder:

1. **Gemeinsam in einem erweiterten Messvektor** verarbeitet werden:

   .. math::

      z_k = \begin{bmatrix} z_k^{(1)} \\ z_k^{(2)} \\ \vdots \\ z_k^{(n)} \end{bmatrix}, \quad
      H = \begin{bmatrix} H^{(1)} \\ H^{(2)} \\ \vdots \\ H^{(n)} \end{bmatrix}, \quad
      R = \begin{bmatrix} R^{(1)} &        &        \\
                              & \ddots &        \\
                              &        & R^{(n)} \end{bmatrix}

   Dies führt zu einem einzigen Update-Schritt mit aggregierter Information.

2. **Sequentiell nacheinander** verarbeitet werden – mit jeweils eigenem Update:

   - Nach jeder Messung wird der Zustand aktualisiert.
   - Der nächste Sensor nutzt den bereits verbesserten Schätzwert als Ausgangspunkt.

**Vergleich und Anwendung**

- Die **gemeinsame Verarbeitung** (Option 1) ist effizienter und optimal unter der Annahme, dass die Messungen unkorreliert sind.
- Die **sequentielle Verarbeitung** (Option 2) ist flexibler und erlaubt z. B. unterschiedliche Messraten oder unsortierten Eingang.

Beide Methoden liefern bei korrekt spezifizierten Modellen und Rauschkovarianzen identische Resultate, solange die Messfehler unabhängig sind.

Das Meß- und Bewegungsmodell in dieser Aufgabe
----------------------------------------------

Für das Beispiel in diesem Praktikum wird ein einfaches lineares Modell mit konstanter Geschwindigkeit verwendet. 
Der Systemzustand :math:`x_k` ist ein 4-dimensionaler Vektor bestehend aus Position und Geschwindigkeit in 2D:

.. math::

   x_k = \begin{bmatrix} x \\ y \\ \dot{x} \\ \dot{y} \end{bmatrix}_k

Dabei sind :math:`x, y` die Positionen und :math:`\dot{x}, \dot{y}` die jeweiligen Geschwindigkeiten in x- und y-Richtung.

**Bewegungsmodell**

Das Bewegungsmodell basiert auf konstanter Geschwindigkeit und ist daher linear. Für eine feste Zeitschrittweite :math:`\Delta t` ergibt sich die Systemmatrix :math:`A`:

.. math::

   A = \begin{bmatrix}
   1 & 0 & \Delta t & 0 \\
   0 & 1 & 0 & \Delta t \\
   0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 1
   \end{bmatrix}

Die Systemdynamik lautet:

.. math::

   x_k = A x_{k-1} + w_{k-1}

mit :math:`w_k \sim \mathcal{N}(0, Q)` als Prozessrauschen.

**Messmodell**

Die Messungen liefern ausschließlich die Position (nicht die Geschwindigkeit) und sind 2-dimensional:

.. math::

   z_k = \begin{bmatrix} x \\ y \end{bmatrix}_k + v_k

Das Messmodell ist ebenfalls linear, mit der Beobachtungsmatrix :math:`H`:

.. math::

   H = \begin{bmatrix}
   1 & 0 & 0 & 0 \\
   0 & 1 & 0 & 0
   \end{bmatrix}

Das Messrauschen :math:`v_k` wird vom Sensor bereitgestellt und ist im Allgemeinen zeitabhängig:

.. math::

   v_k \sim \mathcal{N}(0, R_k)

Dabei ist :math:`R_k` eine 2×2-Kovarianzmatrix, die für jeden Zeitschritt vom Sensor mitgeliefert wird.

Bemerkung:  
Die Struktur dieses Modells erlaubt eine sehr effiziente Anwendung des Kalman-Filters, da beide Modelle linear sind und die Messungen direkt zur Korrektur der Positionsschätzung beitragen.


**Aufgabe 1** - Abhängigkeiten installieren
-------------------------------------------

Installieren Sie zunächst die benötigten Abhängigkeiten, um das Kalman-Filter zu implementieren. 
Wechseln Sie dazu in das Unterverzeichniss `kalman` und führen Sie den Befehl

      pip install -r requirements.txt

aus. Sie implementieren die folgenden Aufgaben in der Datei 

      kalman.py
          

**Aufgabe 2** - Das Filter initialisieren
-----------------------------------------

Das Kalman Filter muss zunächst initialisiert werden, bevor es verwendet werden kann.

Implementieren Sie eine nun die Funktion :py:meth:`kalman.kalman.init_filter`. Folgen Sie den 
Anweisungen im Code sowie dieser Beschreibung.

.. automethod:: kalman.KalmanFilter.init_filter    

**Aufgabe 3** - Die Prädiktion
------------------------------

Bevor das Kalman-Filter neue Messungen verarbeiten kann, muss es den nächsten Zustand vorhersagen.

Implementieren Sie eine nun die Funktion :py:meth:`kalman.kalman.predict`. Folgen Sie den 
Anweisungen im Code sowie dieser Beschreibung.

.. automethod:: kalman.KalmanFilter.predict

**Aufgabe 4** - Die Messungen verarbeiten
-----------------------------------------

Nun kann das Kalman-Filter neue Messungen verarbeiten und den Zustand aktualisieren.

Implementieren Sie eine nun die Funktion :py:meth:`kalman.kalman.update`. Folgen Sie den 
Anweisungen im Code sowie dieser Beschreibung.

.. automethod:: kalman.KalmanFilter.update

Die Simulation starten
----------------------

Um ihr Kalman-Filter zu testen, müssen Sie die Simulation starten.
Führen Sie dazu im Verzeichnis `kalman` den Befehl

      python kalman.py  

aus. Dies startet die Simulation und zeigt die Ergebnisse an.

Die Simulation zeigt die Bewegung eines Objekts in 2D, 
wobei das Kalman-Filter die Position und Geschwindigkeit schätzt. 

Im Hauptfenster sehen Sie die Position des Objekts, die durch das Kalman-Filter geschätzt wird (grau), 
sowie die tatsächliche Position (rot) und die Messungen (blau).

.. image:: simulation1.png
   :width: 800px
   :align: center
   :alt: Kalman-Filter Simulation (Position)

Im Fenster "Geschwindigkeit" sehen Sie die geschätzte Geschwindigkeit des Objekts (grün).

.. image:: velocity1.png
   :width: 800px
   :align: center
   :alt: Kalman-Filter Simulation (Geschwindigkeit)

Sie können die Simulation durch Drücken der Leertaste pausieren und fortsetzen.
Sie können auch die Simulation beenden, indem Sie das Fenster schließen oder die Escape-Taste drücken.
Durch drücken der Taste "+" oder "-" können Sie einzelne Schritte vorwärts oder oder rückwärts gehen.   
Im Menü des Hauptfenster können Sie zusätzliche Visualisierungen aktivieren, 
um die Funktionsweise des Kalman-Filters besser zu verstehen.

Variationen
-----------

In der Standarausführung gibt es einen Sensor, der die Position des Objekts misst.
Es wird ein GPS-Sensor simuliert, der die Position des Objekts in 2D misst.
Dieser Sensor hat eine niedrige Positionsgenauigkeit, fällt dafür aber nur sehr selten aus.
Sie können diese Einstellungen anpassen um zu experimentieren. Öffnen Sie dazu die Datei `config.yml` und passen Sie
die Sektion `sensor` an.

.. code-block:: yaml  
      
      sensors:
      - name: GPS
        type: global
        stdX: 5.0 # Unit: meters
        stdY: 5.0 # Unit: meters
        correlation: 0.0 # between -1 and 1
        missChance: 0.05
        missLength: 5

Sie können auch weitere Sensoren hinzufügen mit anderen Eigenschaften.
Alternativ können Sie auch weitere vorkonfigurierte Szenarien ausprobieren, 
indem Sie das Programm mit einer anderen Konfigurationsdatei starten. Führen Sie dazu den Befehl

      python kalman.py --cfg config_dgps.yaml

aus. Es gibt verschiedene Konfigurationen, die Sie ausprobieren können:

.. code-block:: text

    config_dgps.yml             # Zwei Sensoren: Differential GPS mit hoher Genauigkeit plus normales GPS
    config_radar1.yml           # Ein einzelner Radar
    config_multiple_radars.yml  # Mehrere Radarsensoren
    config_full.yml             # GPS, dGPS plus 4 Radarsensoren

Musterlösung
------------

:doc:`source`

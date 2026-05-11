.. _good_features:

Good Features To Track
======================

In diesem Praktikum verwenden wir die OpenCV-Funktion
`cv2.goodFeaturesToTrack <https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga4055896d9ef77dd3cacf2c5f60e13f1>`_,
um starke und gut voneinander getrennte Ecken in einem Kamerabild zu finden.
Die Funktion baut auf derselben Grundidee wie der Harris-Eckendetektor auf,
nimmt uns aber die Berechnung der Eckenstärke, die Schwellwertbildung und die
Auswahl lokaler Maxima ab.

Theorie
-------

Für jeden Bildpunkt wird lokal der Strukturtensor

.. math::

    M =
    \sum_{(x_k,y_k)\in W}
    \begin{pmatrix}
        I_x^2 & I_x I_y\\
        I_x I_y & I_y^2
    \end{pmatrix}

betrachtet. Seine Eigenwerte :math:`\lambda_1` und :math:`\lambda_2`
beschreiben, wie stark sich die Bildintensität in zwei unabhängigen Richtungen
ändert.

- Ist nur ein Eigenwert groß, liegt typischerweise eine Kante vor.
- Sind beide Eigenwerte groß, liegt typischerweise eine Ecke vor.
- Sind beide Eigenwerte klein, handelt es sich um eine nahezu konstante Fläche.

Die Funktion `cv2.goodFeaturesToTrack` kann entweder das Harris-Kriterium

.. math::

    R = det(M) - \kappa \cdot tr(M)^2

verwenden oder standardmäßig das Shi-Tomasi-Kriterium

.. math::

    R = \min(\lambda_1, \lambda_2)

nutzen. Danach werden nur Punkte behalten, deren Qualität mindestens
`qualityLevel` mal so groß ist wie die beste gefundene Ecke. Zusätzlich sorgt
`minDistance` dafür, dass die zurückgegebenen Punkte nicht zu dicht
beieinander liegen.

Der Code
--------

In diesem Praktikum arbeiten Sie in der Datei

.. code-block:: bash

    harris/good_features.py

Implementieren Sie zunächst die Funktion `processImage`. Diese soll ein
Kamerabild entgegennehmen, gute Ecken bestimmen und ein neues Bild mit
eingezeichneten Markierungen zurückgeben. Anschließend vervollständigen Sie den
Haupt-Loop, so dass die Webcam geöffnet und das Ergebnis live angezeigt wird.

**Aufgabe 1**: Das Bild in Graustufen umwandeln
----------------------------------------------------------------

`cv2.goodFeaturesToTrack` arbeitet auf einem einkanaligen Bild. Wandeln Sie das
Kamerabild deshalb zuerst mit `cv2.cvtColor` in ein Grauwertbild um.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

**Aufgabe 2**: Gute Ecken finden
----------------------------------------------------------------

Rufen Sie nun `cv2.goodFeaturesToTrack` mit dem Grauwertbild auf. Verwenden Sie
zunächst maximal 100 Ecken, ein `qualityLevel` von 0.01 und einen
Mindestabstand von 10 Pixeln zwischen zwei Ecken.

Die Funktion gibt entweder `None` zurück, wenn keine passenden Ecken gefunden
wurden, oder ein Array der Form :math:`N \times 1 \times 2`. Die letzte
Dimension enthält jeweils die Koordinaten :math:`(x,y)` einer Ecke.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
        )

**Aufgabe 3**: Die gefundenen Ecken einzeichnen
----------------------------------------------------------------

Zeichnen Sie die gefundenen Ecken auf einer Kopie des Kamerabildes ein. Prüfen
Sie vorher, ob überhaupt Ecken gefunden wurden. Da OpenCV zum Zeichnen ganze
Pixelkoordinaten erwartet, wandeln Sie die Koordinaten mit `np.int32` um.

Verwenden Sie anschließend `cv2.circle`, um jede Ecke als kleinen grünen Kreis
zu markieren.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        overlay = frame.copy()
        if corners is not None:
            corners = np.int32(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)

        return overlay

**Aufgabe 4**: Die Webcam öffnen und das Ergebnis anzeigen
----------------------------------------------------------------

Vervollständigen Sie nun den Haupt-Loop in `mainLoop`. Öffnen Sie die
Standardkamera mit `cv2.VideoCapture(0)`, lesen Sie in einer Endlosschleife
jeweils ein neues Bild, rufen Sie `processImage` auf und zeigen Sie das Ergebnis
mit `cv2.imshow` an.

Beenden Sie das Programm, wenn der Benutzer die Escape-Taste drückt. Der
zugehörige Tastencode ist 27. Geben Sie am Ende die Kamera wieder frei und
schließen Sie alle OpenCV-Fenster.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        cam = cv2.VideoCapture(0)

        while True:
            _, frame = cam.read()
            overlay = processImage(frame)
            cv2.imshow("Good Features To Track", overlay)

            if cv2.waitKey(1) == 27:
                break

        cam.release()
        cv2.destroyAllWindows()

Experimentieren
---------------

Variieren Sie die Parameter von `cv2.goodFeaturesToTrack` und beobachten Sie
das Ergebnis:

- Erhöhen Sie `maxCorners`, um mehr mögliche Ecken zuzulassen.
- Erhöhen Sie `qualityLevel`, um nur sehr starke Ecken zu behalten.
- Erhöhen Sie `minDistance`, um die gefundenen Ecken weiter auseinander zu
  halten.
- Setzen Sie `useHarrisDetector=True`, um statt des Shi-Tomasi-Kriteriums das
  Harris-Kriterium zu verwenden.

Good Features To Track - Musterlösung
-------------------------------------

:doc:`good_features_source`

.. _opticalflow:

Optischer Fluss mit KLT
=======================

In dieser Praktikumsaufgabe implementieren wir einen einfachen KLT-Tracker
auf dem Livebild der Webcam. KLT steht hier fuer
**Kanade-Lucas-Tomasi**: Lucas und Kanade liefern die lokale
Bewegungsschaetzung, Tomasi bzw. Shi-Tomasi liefert gute Punkte, fuer die
diese Schaetzung numerisch stabil ist.

Wir verwenden zwei OpenCV-Funktionen:

- `cv2.goodFeaturesToTrack <https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga4055896d9ef77dd3cacf2c5f60e13f1>`_
  um gute Ecken im Bild zu finden.
- `cv2.calcOpticalFlowPyrLK <https://docs.opencv.org/4.x/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323>`_
  um diese Punkte von einem Kamerabild ins naechste zu verfolgen.

Optischer Fluss
---------------

Optischer Fluss beschreibt die scheinbare Bewegung von Bildpunkten zwischen
zwei aufeinanderfolgenden Bildern. Wenn sich ein Objekt im Bild bewegt, dann
verschiebt sich seine Projektion auf dem Bildsensor. Diese Verschiebung nennen
wir fuer einen Bildpunkt

.. math::

    \vec d =
    \begin{pmatrix}
        u\\
        v
    \end{pmatrix}

mit horizontaler Bewegung :math:`u` und vertikaler Bewegung :math:`v`.

Die grundlegende Annahme ist die **Helligkeitskonstanz**: Ein kleiner
Bildausschnitt sieht im naechsten Bild noch gleich aus, nur an einer leicht
verschobenen Position.

.. math::

    I(x, y, t) = I(x + u, y + v, t + 1)

Mit einer Taylorapproximation erster Ordnung erhalten wir daraus die bekannte
optical-flow equation:

.. math::

    I_x u + I_y v + I_t = 0

Dabei sind :math:`I_x` und :math:`I_y` die raeumlichen Bildableitungen und
:math:`I_t` ist die zeitliche Aenderung zwischen zwei Frames.

Das Aperturproblem
------------------

Die Gleichung

.. math::

    I_x u + I_y v + I_t = 0

enthaelt zwei unbekannte Groessen, :math:`u` und :math:`v`, aber nur eine
Gleichung. An einer reinen Kante kann man daher nur die Bewegung senkrecht zur
Kante sicher bestimmen. Die Bewegung entlang der Kante bleibt unklar. Dieses
Problem nennt man **Aperturproblem**.

Lucas und Kanade loesen dieses Problem lokal. Die Annahme lautet: In einem
kleinen Fenster :math:`W` um einen Punkt bewegen sich alle Pixel ungefaehr mit
derselben Verschiebung :math:`(u,v)`. Dann koennen wir viele Pixelgleichungen
gemeinsam betrachten und folgenden Fehler minimieren:

.. math::

    E(u, v) =
    \sum_{(x_k,y_k)\in W}
    \left(I_x(x_k,y_k)u + I_y(x_k,y_k)v + I_t(x_k,y_k)\right)^2

Die Normalgleichungen fuer dieses kleinste-Quadrate-Problem lauten:

.. math::

    \begin{pmatrix}
        \sum I_x^2 & \sum I_x I_y\\
        \sum I_x I_y & \sum I_y^2
    \end{pmatrix}
    \begin{pmatrix}
        u\\
        v
    \end{pmatrix}
    =
    -
    \begin{pmatrix}
        \sum I_x I_t\\
        \sum I_y I_t
    \end{pmatrix}

Die Matrix links ist wieder der Strukturtensor aus dem Harris-Kapitel. Genau
deshalb tracken wir bevorzugt Ecken: Bei einer guten Ecke sind beide Eigenwerte
des Strukturtensors gross genug. Die Matrix ist dann gut invertierbar und die
gesuchte Verschiebung :math:`(u,v)` ist stabil bestimmbar.

Pyramidaler Lucas-Kanade
------------------------

Die lokale Lucas-Kanade-Schaetzung funktioniert besonders gut fuer kleine
Bewegungen. In einem Webcam-Bild koennen Bewegungen aber auch groesser als nur
ein oder zwei Pixel sein. OpenCV verwendet deshalb eine Bildpyramide:

- Zuerst wird die Bewegung auf einer kleineren Version des Bildes geschaetzt.
- Diese grobe Schaetzung wird auf der naechstgroesseren Ebene verfeinert.
- Am Ende erhalten wir die Punktposition im originalen Kamerabild.

Diese Variante ist in ``cv2.calcOpticalFlowPyrLK`` bereits implementiert.

Der Code
--------

In diesem Praktikum arbeiten Sie in der Datei

.. code-block:: bash

    opticalflow/klt.py

Die Musterloesung befindet sich in ``opticalflow/klt_solution.py``.

Die Parameter fuer ``cv2.goodFeaturesToTrack`` und
``cv2.calcOpticalFlowPyrLK`` sind im Code bereits als Konstanten
``FEATURE_PARAMS`` und ``LK_PARAMS`` vorgegeben. Auch der Webcam-Loop und die
Zeichenfunktionen sind bereits implementiert. In dieser Aufgabe konzentrieren
wir uns daher nur auf die vier Kernschritte des KLT-Trackings.

**Aufgabe 1**: Das Kamerabild in Graustufen umwandeln
-----------------------------------------------------

Implementieren Sie ``toGray``. Die Funktion erhaelt ein Farbbild von der
Webcam. OpenCV speichert dieses Bild in BGR-Reihenfolge mit der Form
:math:`H \times W \times 3`.

Wandeln Sie das Bild mit ``cv2.cvtColor`` und ``cv2.COLOR_BGR2GRAY`` in ein
einkanaliges Grauwertbild um und geben Sie dieses Bild zurueck.

.. admonition:: Loesung anzeigen
   :class: toggle

   .. code-block:: python

        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

**Aufgabe 2**: Gute Featurepunkte detektieren
---------------------------------------------

Implementieren Sie ``detectFeaturePoints``. Die Funktion erhaelt das aktuelle
Grauwertbild. Rufen Sie ``cv2.goodFeaturesToTrack`` mit diesem Bild auf und
uebergeben Sie die bereits vorbereiteten Parameter ``FEATURE_PARAMS``.

Die Funktion soll die OpenCV-Rueckgabe unveraendert zurueckgeben. Diese ist
entweder ``None`` oder ein Array der Form :math:`N \times 1 \times 2`.

.. admonition:: Loesung anzeigen
   :class: toggle

   .. code-block:: python

        return cv2.goodFeaturesToTrack(gray, **FEATURE_PARAMS)

**Aufgabe 3**: Punkte mit KLT tracken
-------------------------------------

Implementieren Sie ``trackFeaturePoints``. Die Funktion erhaelt das vorherige
Grauwertbild, das aktuelle Grauwertbild und die Punkte aus dem vorherigen Bild.

Falls keine Punkte vorhanden sind, geben Sie ``None, None, None`` zurueck.
Andernfalls rufen Sie ``cv2.calcOpticalFlowPyrLK`` auf. Uebergeben Sie:

- das vorherige Grauwertbild,
- das aktuelle Grauwertbild,
- die vorherigen Punkte,
- ``None`` als Platzhalter fuer die neuen Punkte,
- die vorbereiteten Parameter ``LK_PARAMS``.

Geben Sie alle drei OpenCV-Rueckgabewerte zurueck: neue Punktpositionen,
Status-Flags und Fehlerwerte.

.. admonition:: Loesung anzeigen
   :class: toggle

   .. code-block:: python

        if previousPoints is None or len(previousPoints) == 0:
            return None, None, None

        return cv2.calcOpticalFlowPyrLK(
            previousGray,
            gray,
            previousPoints,
            None,
            **LK_PARAMS,
        )

**Aufgabe 4**: Nur erfolgreiche Tracks behalten
-----------------------------------------------

Implementieren Sie ``keepGoodTracks``. OpenCV liefert fuer jeden Punkt ein
``status``-Flag. Ein Wert von ``1`` bedeutet, dass der Punkt erfolgreich
getrackt wurde. Ein Wert von ``0`` bedeutet, dass der Track verworfen werden
soll.

Wenn ``currentPoints`` oder ``status`` den Wert ``None`` haben, geben Sie zwei
leere Arrays der Form :math:`0 \times 2` zurueck. Andernfalls wandeln Sie
``status`` in eine boolesche Maske um, selektieren damit die erfolgreichen
alten und neuen Punktpositionen und geben beide Arrays mit der Form
:math:`M \times 2` zurueck.

.. admonition:: Loesung anzeigen
   :class: toggle

   .. code-block:: python

        if currentPoints is None or status is None:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        good = status.reshape(-1).astype(bool)
        goodPrevious = previousPoints[good].reshape(-1, 2)
        goodCurrent = currentPoints[good].reshape(-1, 2)

        return goodPrevious, goodCurrent

Experimentieren
---------------

Variieren Sie die Parameter und beobachten Sie die Wirkung:

- Erhoehen Sie ``qualityLevel``, um nur sehr starke Ecken zu tracken.
- Erhoehen Sie ``minDistance``, um die Punkte gleichmaessiger im Bild zu
  verteilen.
- Vergroessern Sie ``winSize``, wenn die Tracks bei etwas Rauschen instabil
  werden.
- Erhoehen Sie ``maxLevel``, wenn Sie groessere Bewegungen verfolgen moechten.

KLT Optischer Fluss - Musterloesung
-----------------------------------

:doc:`source`

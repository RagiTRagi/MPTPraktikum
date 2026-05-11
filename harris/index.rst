Der Harris Eckendetektor
========================

1988 beschrieben Chris Harris und Mike Stephens einen verbesserten 
`Eckendetektor <https://en.wikipedia.org/wiki/Harris_corner_detector>`_ als
Erweiterung des Eckendetektors von `Moravec <https://en.wikipedia.org/wiki/Corner_detection#Moravec_corner_detection_algorithm>`_.

Während Moravec die *Summe der quadratischen Differenzen* 

.. math::
    f(\Delta x, \Delta y)
    =
    \sum_{(x_k, y_k)\in W} \left(
        I(x_k, y_k) - I(x_k + \Delta x, y_k + \Delta y)
    \right)^2

über einem Fenster :math:`W` für vier grundlegende Richtungen 

.. math::
    (\Delta x, \Delta y) \in \{ (1,0), (0, 1), (1,1), (1,-1) \}

betrachtet, erweiteren Harris und Stephens diese Idee. Sie approximieren
:math:`I(x_k + \Delta x, y_k + \Delta y)` über die s.g. `Taylorapproximation <https://en.wikipedia.org/wiki/Taylor_series>`_
ersten Grades, also 

.. math::
    I(x_k + \Delta x, y_k + \Delta y)
    \approx
    I(x_k, y_k) + \Delta x \frac{\partial I}{\partial x} + \Delta y \frac{\partial I}{\partial y}

und erhalten damit die Approximation

.. math::
    f(\Delta x, \Delta y)
    \approx
    \sum_{(x_k, y_k)\in W} \left(
        \Delta x \frac{\partial I}{\partial x} + \Delta y \frac{\partial I}{\partial y}
    \right)^2

Diese Gleichung läßt sich mit 

.. math::
    M = \sum_{(x_k,y_k)\in W}
    \begin{pmatrix}
    I_x^2 & I_x I_y\\
    I_x I_y & I_y^2
    \end{pmatrix}

schreiben als

.. math::
    f(\Delta x, \Delta y)
    \approx
    \begin{pmatrix}
    \Delta x & \Delta y
    \end{pmatrix}
    \cdot M \cdot
    \begin{pmatrix}
    \Delta x \\ \Delta y
    \end{pmatrix}

Dabei nennt man :math:`M` **Strukturtensor**.
Harris und Stephens berechnen dann die s.g. Eckenstärke als 

.. math::

    R = det(M) - \kappa tr(M)^2
    
mit :math:`\kappa \approx 0.04`. 
In diesem Praktikum implementieren wir den Eckendetektor "from scatch".

Der Code
--------

In diesem Praktikum arbeiten Sie in der Datei

.. code-block:: python
    
    kanten.py

und implementieren die Funktion *processImage*

**Schritt 1**: Das Bild in Graustufen umwandeln
-----------------------------------------------
Wir müssen zunächst das Bild in Graustufen umwandeln. Konvertieren Sie 
das Bild mit der `cv2.cvtColor <https://www.geeksforgeeks.org/python-opencv-cv2-cvtcolor-method/>`_ Methode.
Wandeln Sie das Bild danach über `np.float32 <https://numpy.org/doc/stable/user/basics.types.html>`_ in ein Float-Bild um. 
Normieren Sie die Grauwerte vorher indem Sie durch 255.0 teilen.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = np.float32(frame_gray / 255.0)

**Schritt 2**: Die Bildgradienten berechnen
-------------------------------------------

Berechnen Sie nun mit der `cv2.Sobel <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d>`_ Methode wieder die Bildableitungen in X- und Y-Richtung. Verwenden Sie `ksize=3` und `ddepth=cv2.CV_32F`. 

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3) / 4.0
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3) / 4.0

**Schritt 3**: Den Strukturtensor :math:`M` vorbereiten
-------------------------------------------------------

Um den Strukturtensor berechnen zu können berechnen Sie zunächst

.. math::
    \begin{align}
    I_x^2 &=& \left(\frac{\partial I}{\partial x}\right)^2\\
    I_y^2 &=& \left(\frac{\partial I}{\partial y}\right)^2\\
    I_x\cdot I_y &=& \left(\frac{\partial I}{\partial x}\right)\cdot\left(\frac{\partial I}{\partial x}\right)
    \end{align}

und speichern die Ergebnisse in drei neuen Bildern. 

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        Ix2  = gx ** 2
        IxIy = gx * gy
        Iy2  = gy ** 2

**Schritt 4**: Den Strukturtensor berechnen
-------------------------------------------

Der Strukturtensor :math:`M` für jeden einzelnen Pixel ergibt sich durch 
Summation der Bilder :math:`I_x^2`, :math:`I_y^2` sowie :math:`I_x I_y`. Um dies zu erreichen falten wir diese Bilder
mit einem s.g. Block-Kernel

.. math::
    \begin{pmatrix}
    1&\dots&1\\
    \vdots&\ddots&\vdots\\
    1&\dots&1
    \end{pmatrix}

bestehend nur aus einsen. Importieren Sie das `scipy <https://scipy.org/>`_ Paket 
und verwenden Sie die `convolve2d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html>`_ Methode
um diese Faltung durchzuführen. Sie können einen geeigneten Kernel mit `np.ones <https://numpy.org/doc/2.2/reference/generated/numpy.ones.html#>`_ erzeugen. 
**Tip**: Dividieren Sie durch die Anzahl einsen in ihrem Kernel um effektiv der Mittelwert anstatt der Summe zu bilden.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        N = 7
        Ix2  = signal.convolve2d(Ix2, np.ones((N,N))) / (N**2)
        Iy2  = signal.convolve2d(Iy2, np.ones((N,N))) / (N**2)
        IxIy = signal.convolve2d(IxIy, np.ones((N,N))) / (N**2)

**Schritt 5**: Die Eckenstärke berechnen
----------------------------------------

Um die Eckenstärke zu berechnen betrachten wir zunächst die nochmal Gleichung 

.. math:: 
    R = det(M) - tr(M)^2

mit 

.. math::
    M = \sum_{(x_k,y_k)} \begin{pmatrix}
        I_x^2 & I_x I_y \\
        I_x I_y & I_y^2
    \end{pmatrix}

Für die `Determinante <https://de.wikipedia.org/wiki/Determinante>`_ finden wir 

.. math::
    det(M) =  \left(\sum_{(x_k,y_k)} I_x^2\right) \left( \sum_{(x_k,y_k)} I_y^2 \right)
           -  \left( \sum_{(x_k,y_k)} I_x I_y \right)^2 

und für die `Spur <https://de.wikipedia.org/wiki/Spur_(Mathematik)>`_ finden wir entsprechend

.. math::
    tr(M) = \sum_{(x_k,y_k)} I_x^2 + \sum_{(x_k,y_k)} I_y^2

Berechnen Sie nun die Determinante sowie die Spur im Python-Code und berechnen Sie dann die Eckenstärke :math:`R` für :math:`kappa = 0.04`.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        kappa = 0.04
        det = Ix2 * Iy2 - IxIy ** 2
        trace = Ix2 + Iy2
        strength = det - kappa *  trace**2

**Schritt 6**: Die Eckenstärke anzeigen
---------------------------------------

Die Eckenstärke kann sowohl negativ als auch positiv sein. Wir sind jedoch nur an großen positiven Werten interessiert.
Normieren Sie die Eckenstärke, indem Sie durch das globale Maximum dividieren (verwenden Sie `np.max <https://numpy.org/doc/2.2/reference/generated/numpy.max.html>`_).

Nun binarisieren wir das Bild indem wir zunächst ein gleichgroßes Bild erzeugen (z.B. mit `np.zeros_like <https://numpy.org/doc/2.2/reference/generated/numpy.zeros_like.html>`_) 
und überall dort, wo die Eckenstärke größer ist als ein Schwellwert :math:`T` eine 1 setzen. 
Zeigen Sie dann beide Bilder mit `cv2.imshow <https://numpy.org/doc/2.2/reference/generated/numpy.zeros_like.html>`_ an.

.. admonition:: Lösung anzeigen
   :class: toggle

   .. code-block:: python

        strength /= np.max(strength) 
    
        corners = np.zeros_like(strength)
        corners[strength > 0.1] = 1.0   

        cv2.imshow("Harris Corner Strength", strength)
        cv2.imshow("Harris Corners", corners)

Musterlösung
------------

:doc:`source`

Weitere Aufgabe
---------------

:doc:`good_features`

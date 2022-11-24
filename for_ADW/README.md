# Hier ist ein paar Codes und eine kurze Einleitung für Porjekt 1 in der Vorlesung ADW

Hallo Leute, 

Wie in der Vorlesung von Holder erwähnt, werde ich den Teil des maschinellen Lernens (Projekt 1) unterstützen.

In 'main.py' sind Codes für eine einfache Implementierung des Projekts. Wenn du noch keine Idee hast, kannst ausprobieren. Dies ist nur ein Beispielcode, die Leistung sollte schlecht sein.... Für das Testbild schlage ich vor, dass Nutzer statt Kreisen nur Wörter mit schwarzem Stift auf weißes Papier schreiben können. 

Wenn du versuchen möchtest, handgeschriebene Kreise oder in Kreisen gezeichnete Wörter zu erkennen, findest du natürlich auch Methoden, wie z. B. geeignete morphologische Operationen, aber es wird empfohlen mit dem einfachsten Szenario zu beginnen.

Wenn du Anregungen oder Fragen habst, kontaktierst mich einfach via <hui.chen@stud.uni-hannover.de>. Oder sage mir in der Vorlesung, ich helfe dir bei der Vorlesung mit Holger und Philip. Oder Schreiben in 'issue'(egal, ob von deiner Gruppe oder hier) ist auch möglich. Ich werde regelmäßig sehen.

Viel Spaß und Erfolg! :-)

## Eine kurze Einleitung dafür, Objekts (Kreis, Linien...) im Bild zu detektieren

Um den Unterschied zwischen traditionellen und Deep-Learning-Methoden so deutlich wie möglich zu machen, stellen Sie ein einfaches Szenario vor:

Am Tag der Mathe-Klausur möchten Sie zu Hause Computerspiele spielen und den Test nicht ablegen, also möchten Sie, dass ein Roboter den Test für Sie ablegt. Wie lassen Sie den Bot nun wissen, wie er die Fragen (1+1=?) beantworten muss, um die Prüfung zu bestehen? 
- Tranditionelles Verfahren: Teilen Sie der Maschine die Additions- und Subtraktionsregeln mit und lassen Sie den Roboter das Ergebnis streng nach den Regeln erhalten. Roboter müssen in der Klausur diese Regel nur anwenden, ohne nachzudenken.
- Maschinelles Lernen: Geben Sie dem Roboter tausend mathematische Aufgaben, teilen Sie ihm auch die Antwort mit und lassen Sie den Roboter lernen, welche Regeln er anwenden muss, um die richtige Antwort zu erhalten, um den nie zuvor gesehenen Fragen in der Prüfung zu antworten. In der Prüfung soll sich der Roboter an vorheriges Training erinnern und die gewonnenen Erfahrungen nutzen, um die Lösung der Frage zu finden.

Bei einfachen Problemen lassen sich mit traditionellen Methoden sehr gute Ergebnisse erzielen. Aber für komplexere Probleme, wie zum Beispiel, wie man ein Go-Spiel gewinnt, sind die von Menschen definierten Lösungen oft nicht so gut wie die von Maschinen selbst gelernten. 

## Für Kreis- und Lienienerkennung

In diesem Projekt müssen wir Objekte in Bildern erkennen. 

### Tranditionelles Verfahren
Sie können diese Aufgabe einfach so nur mit tradionellen Verfahren mittels OpenCV schaffen:
- Zur Linienerkennung: viele Algorithmuen sind schon in OpenCV zu finden, wie:
    - [Hough-Transform](https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/)
    - [LSD Detector](https://docs.opencv.org/3.4/d1/dbd/classcv_1_1line__descriptor_1_1LSDDetector.html)
    - [FLD Detector](https://docs.opencv.org/4.x/df/ded/group__ximgproc__fast__line__detector.html)
    - ...

Es gibt fast keinen Unterschied in der Leistung, Sie können eine beliebige auswählen. Entsprechende Beispiele befinden sich unten.

- Zur Kreis- und Ellipsenerkennung:
    - [Hough Circles](https://www.geeksforgeeks.org/circle-detection-using-opencv-python/)
    - [scikit-image](https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html#sphx-glr-auto-examples-edges-plot-circular-elliptical-hough-transform-py)
    - [mittels Bloberkennung](https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/)

Ich bin mit Kreiserkennungsalgorithmen nicht sehr vertraut, möglicherweise gibt es andere Algorithmen, die zum Erkennen von Kreisen verwendet werden können. 
Wenn Ihnen die obige Methode nicht gefällt, können Sie natürlich nach Ihrem bevorzugten Algorithmus suchen. 

- Morpologische Operationen
    - [dilate](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)
    - [erode](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)

### Deep Learning
Außerdem ist die Detektion auch mittels Maschinellen Lernens bzw. Deep Learning möglich. Die Hauptschwierigkeit dieser Methode liegt in der Erstellung des Datensatzes und der Erstellung des Modells. 

- Datensatz erstellen. Es gibt viele Varianten von handgeschriebenen Kreisen und Linien, damit das Modell schließlich möglichst alle Varianten erkennt, sollte ein Datensatz erstellt werden, der möglichst viele Änderungen und entsprechende korrekte Annotationen enthält.

- Modell auswählen. Es wird empfohlen, einige bekannte Modelle zu verwenden und ihre vortrainierten Parameter zu laden. Fahren Sie dann mit dem Training an Ihrem eigenen Datensatz fort. 

- Loss Function auswählen. Die Verlustfunktion kann überwachen, ob die vom Modell während des Trainingsprozesses erlernte Problemlösungsstrategie gut ist. Es sollte darauf geachtet werden. 

[Infos für die Unterschiede zwischen semantische Segmentierung und Objekterkennung](https://cs.stackexchange.com/questions/51387/what-is-the-difference-between-object-detection-semantic-segmentation-and-local#:~:text=%22Object%20detection%22%20is%20localizing%20%2B%20classifying%20all%20instances,per-pixel%20classification.%20Also%20wrt%20involved%20metrics%20%28source%3A%20https%3A%2F%2Fdevblogs.nvidia.com%2Fparallelforall%2Fdeep-learning-object-detection-digits%2F%29) 

Für mehr Theorie und Praxis des Deep Learning empfehle ich [den Kurs von Andrew Ng](https://www.deeplearning.ai/courses/). 

Tutorial von EasyOCR -> [How to train your custom model](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md)

## OCR

Es gibt bereits etablierte Algorithmen zur optischen Zeichenerkennung:
- Tesseract -> [Guide](https://nanonets.com/blog/ocr-with-tesseract/), [Github](https://github.com/tesseract-ocr/tesseract)
- Easyocr -> [Guide](https://pyimagesearch.com/2020/09/14/getting-started-with-easyocr-for-optical-character-recognition/), [Github](https://github.com/JaidedAI/EasyOCR)

## Schwierigkeiten

- Das Erkennen von Standardkreisen und -linien kann leicht mit traditionellen Methoden durchgeführt werden, aber handgeschriebene Kreise und Linien sind oft nicht standardisiert.

- Nur wenn Sie über einen geeigneten Datensatz verfügen und ein geeignetes Modell verwenden, könnten Sie mit Deep-Learning-Methoden gute Ergebnisse erzielen. Modelle sind relativ einfach zu finden, aber Datensätze sind schwer.

## Tipps

- Ich empfehle, zuerst die traditionellen Linien- und Kreiserkennungsalgorithmen in OpenCV auszuprobieren. Wenn guter Datensatz vorhanden sind ([MNIST](http://yann.lecun.com/exdb/mnist/), ggf. [hier](https://github.com/clovaai/deep-text-recognition-benchmark), für mehr Details seh [EasyOCR](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md)), können Sie den Deep-Learning ausprobieren. 

- Für die Erkennung von nicht standardmäßigen Kreisen stehen nur wenige Algorithmen zur Verfügung. Ich habe Blob-Erkennungsalgorithmen('blobs.py', [Quelle](https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/)) und Ellipsen-Erkennungsalgorithmen('ellipse.py', [Quelle](https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html#sphx-glr-auto-examples-edges-plot-circular-elliptical-hough-transform-py)) gefunden und hoffe Ihnen ein wenig helfen zu können.

- Ich denke, dass das Erkennen von handschriftlichen Kreisen übersprungen werden kann, weil der Zweck der Erkennung des Kreises nur darin besteht, die Position des Textes weiter zu ermitteln. Einige OCR-Pakete in Python (wie EasyOCR) können nicht nur den Textinhalt im Bild ausgeben, sondern auch gleichzeitig deren Position. Sie können also versuchen, die Textkoordinaten direkt über OCR zu generieren. 

- Mittels tesseract-OCR kann man nicht die Position von Textblöcken bestimmen, deswegen muss man vor OCR mittels tesseract zuerst ROI (region of interesse) abschneiden.

- Wenn sie ihr selbst Modell trainieren wollen -> [How to train your custom model](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md)

- Andere maschinelle Lernalgorithmen (ggf. [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)) neben Deep Learning können diese Aufgabe möglicherweise auch lösen, ich weiß aber wenig darüber und kann keine Tipps geben. 

## Ende
In 'OCR.py' sind Beispielcodes von den zwei OCR-Techniken zu finden. Außerdem sind Codes für Linien- und Kreiserkennung mittels OpenCV in 'detektion.py'. In 'modell.py' ist zwei Modelle für semantische Segmentation sowie die Codes zum Training. Bitte beachten Sie, dass die Codes für Training nur die wichtigsten Teile enthalten, für besseren Code siehe [hier](https://github.com/asagar60/TableNet-pytorch/tree/main/Training). In 'ellipse.py' werden die Codes zur Detektion von Eillipse gebietet, die Quelle ist [hier](https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html#sphx-glr-auto-examples-edges-plot-circular-elliptical-hough-transform-py). Wenn sich in Noden Polsterung befindet, könnten Sie möglicherweise versuchen den Noden zu erkennen, indem Sie sie als Blobs ('blobs.py', [Quelle](https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/)) sehen. In 'morpholo.py' sind Codes für morphologische Operationen. 

Außerdem -> [How to train your custom model](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md), [MNIST](http://yann.lecun.com/exdb/mnist/)

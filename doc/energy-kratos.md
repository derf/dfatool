Diese Anleitung beschreibt die Generierung von Energiemodellen mit dfatool und
Kratos. Sie geht von der folgenden Verzeichnisstruktur aus.

* `data`: Benchmark-Messdaten
* `data/cache`: Cache für teilweise ausgewertete Benchmarks
* `dfatool`: dfatool-Repository
* `kratos`: Kratos-Repository

*kratos* enthält Gerätetreiber mit zugehörigen PTA-Definitionen
(Transitionen, Zustände und Parameter der Hardware) sowie Hilfsfunktionen für
Benchmarks. In *dfatool* liegen die Generierungs- und Auswertungsskripte.

## Benchmarkgenerierung

Die Generierung und Vermessung von Benchmarks erfolgt immer mit
`generate-dfa-benchmark.py`. Dieses muss vom Kratos-Verzeichnis aus
aufgerufen werden. Kratos muss zuvor passend konfiguriert worden sein.

Konfiguration:

* msp430fr5994 "UART on eUSCI\_A" aktiv: "uart" mit 9600 Baud auf A1
* msp430fr5994 "Timer/Counter" aktiv: 16/1/1 == 16 MHz CONTINUOUS von SMCLK auf A1
* msp430fr5994 "Cycle Counter" aktiv: via A1
* apps "dfatool energy benchmark" aktiv (und sonst nix)

Ablauf: Siehe energy-multipass.md

## Beispiel

Wenn sich msp430-etv und energytrace in $PATH befinden und ein CC1101 Funkchip
angeschlossen und in Kratos konfiguriert ist, generiert der folgende Aufruf mit
einem MSP430FR5994 Launchpad einen erfolgreichen Benchmark-Ablauf:

```
cd kratos
../dfatool/bin/generate-dfa-benchmark.py --data=../data \
--os=kratos --sleep=500 --repeat=1 --depth=3 --arch=msp430fr \
--energytrace=sync=timer model/drivers/radio/cc1101_simpliciti.dfa  src/apps/AEMR/AEMR.cc
```

Nach einigen Minuten wird unter `data` ein auf radio.tar endendes Archiv mit
Benchmark-Setup (Treiber-PTA, energytrace-Config, Traces durch den
Automaten) und Messdaten (energytrace-Logfiles) abgelegt. Dieses kann wie folgt
analysiert werden:

```
cd dfatool
bin/analyze-archive.py --info --show-model=all --show-quality=table ../data/...-radio.tar
```

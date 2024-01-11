Diese Anleitung beschreibt die Generierung von Energiemodellen mit dfatool und
multipass. Sie geht von der folgenden Verzeichnisstruktur aus.

* `data`: Benchmark-Messdaten
* `data/cache`: Cache für teilweise ausgewertete Benchmarks
* `dfatool`: dfatool-Repository
* `multipass`: multipass-Repository

*multipass* enthält Gerätetreiber mit zugehörigen PTA-Definitionen
(Transitionen, Zustände und Parameter der Hardware) sowie Hilfsfunktionen für
Benchmarks. Es verzichtet bewusst auf Tasking und System-Ticks, um Benchmarks
nicht durch Timer Interrupts zu beeinflussen. In *dfatool* liegen die
Generierungs- und Auswertungsskripte.

## Benchmarkgenerierung

Die Generierung und Vermessung von Benchmarks erfolgt immer mit
`generate-dfa-benchmark.py`. Dieses muss vom multipass-Verzeichnis aus
aufgerufen werden. Die multipass-Konfiguration (Treiber und Anwendungen)
sowie Codegenerierung läuft automatisch.

* Generierung von Läufen durch den PTA des zu vermessenden Geräts. Die Läufe
  können u.a. mit `--depth`, `--shrink` und `--trace-filter` beeinflusst
  werden.
* Erzeugung einer C++-Anwendung (`src/app/aemr/main.cc`), welche die Hardware
  durch die Läufe schickt und die ausgeführten Transitionen protokolliert. Sie
  greift auf `include/object/ptalog.h` zurück.
  * Die grundlegende Anwendungsstruktur (Header, Aufruf der Treiberfunktionen,
    Wartezeit zwischen Funktionsaufrufen) wird von generate-dfa-benchmark
    vorgegeben (`benchmark_from_runs`)
  * Ein Test Harness aus `lib/harness.py` (OnboardTimerHarness für
    energytrace/timing benchmarks, TransitionHarness für MIMOSA) erweitert
    die generierte Anwendung um Synchronisierungsaufrufe und/oder zusätzliche
    Messungen, z.B. mit einem Onboard-Timer. Dazu werden für jeden Lauf durch
    den PTA `start_run` und `start_trace` aufgerufen ("ein neuer Lauf beginnt"),
    dann für jeden Funktionsaufruf und jeden Zustand `append_transition`,
    `append_state` und `pass_transition` und schließlich `stop_run`.
    Das Harness speichert die zum generierten Code gehörenden Läufe und die
    während eines Zustands / einer Transition gültigen PTA-Parameter intern als
    `{"isa": "state", "name": ..., "parameter": dict(...)}` bzw.
    `{"isa": "transition", "name": ..., "parameter: dict(...), "args": list(...)}`
* Kompilieren der Anwendung in `run_benchmark` per `runner.build` (siehe
  `runner.py`). Falls der Benchmark zu groß ist, wird er in mehrere
  Anwendungen aufgeteilt, die nacheinander ausgeführt und vermessen werden.
  Zusätzlich wird jede Messung mehrfach durchegführt, um Einflüsse durch
  Messfehler zu minimieren.
* Ausführung des Benchmarks. Der Code wird mittels `runner.flash` programmiert,
  die Ansteuerung zusätzlicher Software (z.B. MIMOSA, EnergyTrace) erfolgt über
  einen Monitor aus `lib/runner.py`. Sobald der Monitor mittels `get_monitor`
  erzeugt wird, beginnt die Messung. Während der Messung werden Ausgaben
  von der seriellen Konsole über den `parser_cb` des aktiven Test Harness
  verarbeitet; auf diese Weise wird auch das Ende des Benchmarks erkannt.
  `monitor.close()` beendet die Messung.
* Nach Abschluss aller (Teil)benchmarks und Wiederholungen werden
  die Benchmarkpläne (`harness.traces`), UART-Ausgaben (`monitor.get_lines()`)
  und ggf. zusätzliche Logfiles (`monitor.get_files()`) in eine tar-Datei
  archiviert.

## Beispiel

Wenn sich msp430-etv und energytrace in $PATH befinden, generiert der folgende
Aufruf mit einem MSP430FR5994 Launchpad ohne Peripherie einen erfolgreichen
Benchmark-Ablauf:

```
cd multipass
../dfatool/bin/generate-dfa-benchmark.py --data=../data \
--sleep=50 --repeat=3 --arch=msp430fr5994lp \
--energytrace=sync=timer model/driver/sharp96.dfa src/app/aemr/main.cc
```

Nach einigen Minuten wird unter `data` ein auf sharp96.tar endendes Archiv mit
Benchmark-Setup (Treiber-PTA, energytrace-Config, Traces durch den
Automaten) und Messdaten (energytrace-Logfiles) abgelegt. Dieses kann wie folgt
analysiert werden:

```
cd dfatool
bin/analyze-archive.py --info --show-model=all --show-model-error ../data/...-sharp96.tar
```

Sofern sich die LED-Leistungsaufnahme des verwendeten Launchpads im üblichen
Rahmen bewegt, funktioniert die Auswertung.  Hier sollten für POWEROFF und
POWERON sehr ähnliche Werte herauskommen (da ja keine Peripherie angeschlossen
war) und die writeLine-Transition deutlich mehr Zeit als die restlichen
benötigen.

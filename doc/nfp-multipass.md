Dieses Dokument beschreibt die Generierung von NFP-Modellen für
Softwareprojekte auf Kconfig-Basis.

## Anforderungen

Im Hauptverzeichnes (von dem aus die Build-Befehle aufgerufen werden) müssen
sich die Dateien Kconfig und .config befinden. Das Projekt muss die folgenden
Arten von Befehlen unterstützen:

* make
* make cleaan
* make nfpvalues
* Optional: make randconfig

Die konkreten Befehle können über Optionen von bin/explore-kconfig.py
eingestellt werden.

für multipass. NFP
beziehen sich hierbei auf nichtfunktionale Attribute des erzeugten Systemimages,
wie z.B. ROM- oder RAM-Bedarf.

Frickel-Level: Signifikant.

High-Level:

* In /tmp/tmpfs/multipass befindet sich multipas
* Konfigurationen und zugehörige NFP-Daten landen in /tmp/tmpfs/multipass-data
* var/ess/multipass-model zur state space exploration
* dfatool/bin/analyze-config zur Modellgenerierung

Low-Level:

Es wird viel kompiliert. `/tmp/tmpfs` sollte sich auf einem tmpfs befinden.

```
rsync ~/var/projects/multipass/ /tmp/tmpfs/multipass/
mkcd /tmp/tmpfs/multipass-data
for i in {1..viele}; do echo $i; ~/var/ess/multipass-model/random-romram.py; done
```

Anschließend in dfatool:

```
bin/analyze-config.py /tmp/tmpfs/multipass/Kconfig /tmp/tmpfs/mulitpass-data
```

Das Modell wird nach kconfigmodel.py geschrieben

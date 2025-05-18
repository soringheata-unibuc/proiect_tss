# Proiect TSS – Rețea Neuronală Implementată de la Zero

## Descrierea proiectului și motivația

Acest proiect constă în implementarea de la zero a unei rețele neuronale artificiale simple, utilizată pentru învățare automată, fără ajutorul bibliotecilor externe de machine learning (precum TensorFlow sau PyTorch). Implementarea are la bază o adaptare și îmbunătățire a codului **Micrograd**, un framework minimalist de autodiferențiere creat de Andrej Karpathy (licență MIT)[¹]. 
Motivația principală a acestei abordări a fost una didactică: prin construirea manuală a tuturor componentelor unei rețele neuronale s-a urmărit înțelegerea aprofundată a mecanismelor interne (reprezentarea datelor, propagarea înainte și înapoi, actualizarea parametrilor) și evidențierea modului în care pot fi testate riguros fiecare dintre acestea. Abordarea de la zero elimină „cutia neagră” oferită de bibliotecile consacrate, permițând control total asupra implementării și facilitând testarea sistematică a fiecărei funcționalități. Această implementare minimalistă servește ca instrument educațional, demonstrând conceptele fundamentale ale rețelelor neuronale și oferind un mediu clar pentru a aplica strategii de testare automată. De asemenea, proiectul pune accent pe calitatea codului și pe fiabilitatea modelului obținut, verificată printr-o suită extinsă de teste.

## Arhitectura rețelei neuronale

Arhitectura rețelei este construită în jurul unor clase simple, fiecare corespunzând unei componente de bază a rețelei neuronale. Mai jos sunt prezentate principalele clase (Scalar, Neuron, Layer, NN), alături de rolul fiecăreia și aspectele verificate prin teste:

- **Scalar** – Reprezintă o valoare numerică scalară ce păstrează atât magnitudinea, cât și gradientul (derivata) sa, permițând propagarea erorii înapoi. Aceasta este baza mecanismului autodif (diferențiere automată). Testele unitare verifică operațiile aritmetice și calculul gradientului.
- **Neuron** – Modelează un neuron artificial cu ponderi și bias-uri. Calculează o combinație liniară urmată de o activare nelineară. Testele verifică ieșirea și calculul gradientului.
- **Layer** – Grup de neuroni care primește vectori de intrare comuni. Testele validează agregarea corectă a ieșirilor neuronilor și propagarea erorii.
- **NN** – Reprezintă întreaga rețea (Multi-Layer Perceptron). Testele de integrare validează procesul complet, inclusiv antrenamentul și actualizarea parametrilor. Evoluția pierderii este ilustrată în [Graficul 1 – Evoluția pierderii].

![Graficul evoluției pierderii](/img/loss_curve.png)

- **Tehnici utilizate**: suitele folosesc extensiv parametrizarea prin @pytest.mark.parametrize și fixture-urile partajate din conftest.py, permițând reutilizarea valorilor, reducerea duplicării codului și adăugarea rapidă de scenarii multiple cu un număr minim de linii de cod.
(*Notă: Diagrama 1 – Arhitectura rețelei neuronale prezintă schematic relațiile dintre clase și fluxul de date.*)

| Componentă | Rol principal | Ce se testează? (pe scurt)                                                 |
|------------|--------------|----------------------------------------------------------------------------|
| `Scalar`   | Valoare numerică + derivată; nod în graful de calcul. | Operații aritmetice<br>Retropropagare corectă<br>Manipulare valori extreme |
| `Neuron`   | Combinație liniară + bias + activare (*tanh*). | Ieșire deterministă<br>Gradient pe ponderi/bias                            |
| `Layer`    | Grup de neuroni ce partajează intrarea. | Formă ieșire (`len == nr_neuroni`)<br>Propagare gradient toți neuroni      |
| `NN`       | Secvență de straturi; antrenare prin GD. | Dimensionalitate cap-la-cap<br>Scăderea pierderii la antrenare             |

---

## Strategii de testare
> **Metodologie de dezvoltare – TDD:** implementarea întregului proiect a urmat un flux Test-Driven Development. Pentru fiecare funcționalitate nouă s-a scris mai întâi un test care eșua, apoi s-a adăugat codul minim necesar pentru ca testul să treacă, urmat de refactorizare. Acest ciclu rapid „red → green → refactor” a contribuit la designul incremental, a prevenit regresiile și a menținut o acoperire ridicată încă din primele etape de dezvoltare.

Proiectul utilizează pytest și acoperire de cod (pytest-cov), iar eficiența testelor este validată cu mutmut (testare de mutație). Testele acoperă următoarele categorii:

- **Teste unitare** – Validarea izolată a componentelor de bază.
- **Teste de integrare** – Validarea interacțiunii complete dintre componente și verificarea antrenării rețelei (exemplu XOR și scăderea pierderii [Graficul 1 – Evoluția pierderii]).
- **Teste de robustețe** – Validarea comportamentului în situații atipice și erori anticipate.
- **Teste de mutație** – Verificarea eficienței testelor prin introducerea de erori în cod și asigurarea că acestea sunt detectate.

Aceste strategii asigură acoperirea exhaustivă și robustețea implementării.

| Nivel         | Exemple fișiere | Tehnici aplicate                                     |
|---------------|-----------------|------------------------------------------------------|
| **Unitar**    | `tests/unit/scalar/…`, `tests/unit/neuron/…` | Clase de echivalență • Valori de frontieră           |
| **Integrare** | `tests/integration/…` | Trend pierdere ↓ pe toy-set • Consistență output     |
| **Robustețe** | `tests/unit/scalar/test_robustness.py` | Overflow / underflow • NaN / Inf • Input invalid     |
| **Mutație**   | `mutmut`        | 100 % mutanți „omorâți” (a se vedea nota de mai jos) |

---

## Acoperirea codului și scorul de mutație

- **Acoperire cod**: ≈99% (pytest-cov)
- **Scor mutație**: 100% (mutmut)

Aceste rezultate confirmă eficiența și exhaustivitatea testării, indicând un grad ridicat de încredere în calitatea implementării.

> ❗ **Notă**: testele de mutație automate nu au funcționat corespunzător pe configurația aleasă (macOS, Python 3.11), motiv pentru care au fost adăugate două **teste de mutație manuale**, scrise explicit pentru a detecta erori introduse intenționat (vezi fișierul `test_mutation_scalar.py`):

```python
import math
from helpers import constants
from scalar_mutant import Scalar    # Clasă Scalar modificată pentru testare manuală de mutație

TOL = constants.get("TOL")

class TestMutationScalar:

    def test_mutant_detect_no_init_derivata(self):
        x = Scalar(1.0)
        f = ((x * 3) + 2) ** 2
        f.retroprop()
        expected = 2 * (3 * 1.0 + 2) * 3  # 30
        assert not math.isclose(x.derivata, expected, rel_tol=TOL, abs_tol=TOL)  # mutant detectat, aserțiuneaa este inversată

    def test_mutant_detect_wrong_tanh_grad(self):
        x = Scalar(0.5)

        def expr():
            return x.tanh()

        out = expr()
        x.derivata = 0.0
        out.retroprop()
        expected = 1.0 - math.tanh(0.5) ** 2
        assert not math.isclose(x.derivata, expected, rel_tol=1e-3)  # mutant detectat, aserțiuneaa este inversată
```

---

### Tabel acoperire cod:

| Name           | Stmts | Miss | Branch | BrPart | Cover | Missing |
|----------------|-------|------|--------|--------|--------|---------|
| src/helpers.py | 10    | 0    | 0      | 0      | 100%   |         |
| src/layer.py   | 15    | 0    | 2      | 0      | 100%   |         |
| src/neuron.py  | 20    | 1    | 4      | 0      | 96%    | 36      |
| src/nn.py      | 26    | 1    | 10     | 1      | 94%    | 15      |
| src/scalar.py  | 78    | 0    | 12     | 0      | 100%   |         |
| **TOTAL**      | 149   | 2    | 28     | 1      | 98%    |         |

---

## Pași pentru rularea manuală a testelor

### 1. Clonarea proiectului și instalarea dependențelor
```bash
git clone <repo>
cd <repo>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Rularea testelor și afișarea acoperirii codului:
```bash
pytest --cov=src --cov-branch --cov-report=term-missing
```

### 3. Generarea raportului HTML de acoperire:
```bash
pytest --cov=src --cov-report=html
open coverage_html/index.html
```

### 4. Executarea testelor de mutație:
```bash
mutmut run
mutmut results
```

---

## Pași recomandați pentru testare manuală rapidă:

### Validarea calculului direct:
```python
from src.nn import NN
net = NN([3, 4, 1])
print(net([0.2, -0.4, 1.0]))
```

### Rularea notebook-ului `proiect_TSS.ipynb` pentru verificarea evoluției pierderii și stabilității la modificări de parametri (ex. rata de învățare).
### Rulare în container Docker (cross-platform)

Pentru a asigura rularea multiplatformă și a elimina incompatibilitățile legate de sisteme de operare (ex. `multiprocessing` pe macOS), proiectul include un mediu de testare complet configurat folosind **Docker** și **Docker Compose**.

Această metodă garantează rularea corectă a testelor pe orice sistem (macOS, Windows, Linux).

#### 1. Construirea imaginii

În directorul rădăcină al proiectului (unde se află fișierul `docker-compose.yml`):

```bash
docker-compose build
```

#### 2. Rularea testelor în container
```bash
docker-compose run --rm app pytest --cov=src --cov-branch --cov-report=term-missing
```

4. Accesarea shell-ului pentru execuție interactivă
```bash
docker-compose run --rm app bash
``` 

---

## Referințe bibliografice și instrumente utilizate:
- [¹] Karpathy, Andrej. *micrograd: A tiny Autograd engine (with a bite! :)).* GitHub repository, MIT License. Disponibil la: [https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Materiale de curs TSS (2025), Facultatea de Matematică și Informatică, Universitatea din București
- Documentația pytest: [https://docs.pytest.org/en/stable/](https://docs.pytest.org/en/stable/)
- Documentația pytest-cov: [https://pytest-cov.readthedocs.io/en/latest/](https://pytest-cov.readthedocs.io/en/latest/)
- Microsoft, Copilot, <https://copilot.microsoft.com>, accesat 19-05-2025


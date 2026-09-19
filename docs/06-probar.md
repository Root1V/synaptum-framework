# Probar

**Construir un agente no debería costar dinero ni depender de tener acceso a un modelo.** En CI, en
una máquina sin GPU, o con un despliegue cuya única puerta está gobernada, la alternativa real no es
«modelo local o modelo cloud»: es modelo simulado o nada.

Por eso los dobles son infraestructura de primera clase, no una utilidad de test.

## `FakeGateway` — un guion

```python
from synaptum.testing import FakeGateway, calls, says

gateway = FakeGateway(
    calls("leer", path="/x"),      # el modelo pide una herramienta
    says("el fichero tiene 12 líneas"),
    tools=[leer],
)
```

**No es un mock.** Las herramientas se ejecutan de verdad, el streaming es de verdad, el `Usage` se
calcula. Lo único que no hace es inferir.

El guion se consume en orden y admite más que respuestas:

| Elemento | Qué hace |
|---|---|
| `says("...")` / `calls(...)` | Respuesta guionizada |
| `str` | Atajo de `says` |
| `BaseException` | Se lanza — con `ProviderError` ejercitas los reintentos |
| `Decision` | Se lanza como `Denied`: las tres disposiciones sin montar un motor de políticas |
| `callable(request)` | Responde **según lo que el bucle acaba de mandar** |

Lo último importa más de lo que parece. Un guion posicional responde lo mismo la segunda vez, así
que al reanudar vuelve a pedir la herramienta que ya se ejecutó — y el ejemplo de durabilidad
mediría lo contrario de lo que afirma:

```python
def responder(peticion):
    ya_consultado = any(m.role is Role.TOOL for m in peticion.messages)
    return says("el saldo es 4.200 €") if ya_consultado else calls("consultar_saldo", cuenta="ES91")
```

### Comprobar la cancelación

```python
gateway.chunks_emitted   # cuántos fragmentos llegó a producir
gateway.cancelled        # si supo que lo cerraron
```

Y una advertencia que vale para cualquier medida de cancelación, no solo aquí: **compruébalo dentro
del bucle de eventos**, justo después de cerrar. Al terminar, Python finaliza los generadores
asíncronos vivos, así que comprobarlo después da verde aunque la señal llegue tarde — y en un
proceso que no termina, tarde es igual que nunca.

## `ReplayGateway` — respuestas grabadas

```python
from synaptum.testing import ReplayGateway

gateway = ReplayGateway("fixtures/chat_completion.json", tools=[leer])
```

Los cuerpos grabados pasan por **el adaptador real**, no por un atajo: al bucle llega lo que llegaría
en producción.

**Un guion a mano dice lo que uno espera; una grabación dice lo que el proveedor hizo** — y la
diferencia aparece en los caminos que nadie escribe porque no se le ocurren. Una grabación real
encontró que un stream que era *solo razonamiento* no emitía ni un evento, porque el cuerpo inventado
que había antes solo llevaba deltas de texto.

## Lo que un doble no puede enseñarte

Esto es lo importante de esta página. Los dobles cubren mucho y hay una clase de fallo que **no**
pueden mostrar, porque su comportamiento es demasiado limpio:

- **Un proveedor que descarta un campo** y sigue. La salida estructurada dejaba de llegar y el error
  culpaba al JSON.
- **Un error que tarda.** Un doble devuelve sus errores al instante, así que un bucle que reintenta
  sin esperar parece correcto.
- **Un modelo que no sigue instrucciones.** Dos ejemplos afirmaban conductas del modelo creyendo que
  afirmaban propiedades del framework; con el doble pasaban siempre.
- **Un servidor que sanea sus mensajes de error.**

> **Un ejemplo que solo se ha ejecutado contra un guion no está ejecutado.** Corre lo que escribas
> contra un modelo real al menos una vez antes de creerte lo que afirma.

## Probar tus propios agentes

```python
import pytest
from synaptum import Agent, Session
from synaptum.testing import FakeGateway, calls, says

@pytest.mark.asyncio
async def test_pide_el_saldo_antes_de_responder():
    gateway = FakeGateway(calls("saldo", cuenta="ES91"), says("4.200 €"), tools=[saldo])
    pasos = [p async for p in mi_agente.run("¿cuánto hay?", session=Session("t1", gateway))]

    herramientas = [p.call.name for p in pasos if p.kind == "tool" and p.call]
    assert herramientas == ["saldo"]
    assert pasos[-1].output == "4.200 €"
```

Probar es iterar una lista. No hace falta arrancar nada.

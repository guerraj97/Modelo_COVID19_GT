# Modelo COVID19 GT
Este es un modelo utilizando el SEIR para ver el comportamiento del Coronavirus en Guatemala utilizando datos del MSPAS
Para esto se utilizo el lenguaje de Python, el solver de ecuaciones diferenciales "solve_ivp" y el método de mínimos cuadrados para la aproximación del modelo a los datos.

Se realizaron dos estimaciones, una con 1 millón de habitantes y otra con 17 millones, con la finalidad de comparar ambos resultados. 

El modelo de 1M es un modelo más ajustado a los datos, por lo que las proyecciones son mejores, pero el modelo de 17M muestra un caso un poco más real, aunque bastante alejado de los datos presentados por lo que el modelo puede diferir de los resultados reales.

En ambos casos, se repite que esto es solo un estimado y que puede cambiar debido a diferentes factores que el Modelo SEIR no toma en cuenta, además que las estimaciones iniciales pueden variar, dando un ajuste diferente al modelo.

Se incluye además un modelo SEIRD que toma en cuenta las defunciones, esto con el fin de hacer una comparativa, sin embargo, estas gráficas no se incluyen en este repositorio.

Para quien lo desee, sientase en la libertad de usar, modificar y aplicar este código, dando créditos a este autor. Cualquier otra duda, comentario o recomendación favor escribir a: gue16242@uvg.edu.gt

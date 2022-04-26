from copy import deepcopy

import numpy as np


class MultiLayerPerceptron:

    def __init__(self, tam_vector_entrada, num_neuranas_capas_ocultas, num_neuronas_salida, rango_normalizacion=None):
        self.tam_vector_entrada = tam_vector_entrada
        self.rango_normalizacion = rango_normalizacion if rango_normalizacion is not None else [-5, 5]
        self.num_neuranas_capas_ocultas = [x for x in num_neuranas_capas_ocultas if x > 0]
        self.num_neuronas_salida = num_neuronas_salida
        self.pesos = []
        self.a = []
        self.n = []
        self.s = []
        self.derivadas = []
        self.error_alcanzado_QP = None
        self.ultimo_gradiente = None
        self.ultima_nabla_w = None
        self.lote=None

    def init_pesos(self):
        def obtener_pesos_dentro_del_rango_normalizacion(size):
            rng = np.random.default_rng()
            a = self.rango_normalizacion[0]
            b = self.rango_normalizacion[1]
            diff = 0.000001
            return (b - a) * rng.random(size) + (a + diff if a < 0 else a - diff)

        capas = self.num_neuranas_capas_ocultas  + [self.num_neuronas_salida]
        self.pesos.append(obtener_pesos_dentro_del_rango_normalizacion((capas[0], self.tam_vector_entrada + 1)))
        for i in range(1, len(capas)):
            self.pesos.append(obtener_pesos_dentro_del_rango_normalizacion((capas[i], capas[i-1]  + 1)))
        #print(self.pesos)


    def forward_propagate(self, entradas):
        self.a = []
        self.n = []
        valores_activacion = entradas
        for w in self.pesos:
            valores_activacion = np.insert(valores_activacion, 0, -1)
            
            valores_activacion = valores_activacion.reshape((valores_activacion.shape[0], 1))
            
            self.a.append(valores_activacion)
            net_entradas = np.dot(w, valores_activacion)
            self.n.append(net_entradas)
            valores_activacion = self._funcion_de_activacion(net_entradas)
        self.a.append(valores_activacion)
        return valores_activacion


    def back_propagate(self, error):
        self.s = []
        self.derivadas = []
        num_capas = len(self.a)
        capa_salida = num_capas - 1
        for i in reversed(range(1, num_capas)):
            is_capa_salida = i == capa_salida
            a = self.a[i] if is_capa_salida else np.delete(self.a[i], 0, 0)
            d = self._funcion_de_activacion_derivada(a)
            derivadas = np.diag(d.reshape((d.shape[0],)))
            self.derivadas = [derivadas] + self.derivadas
            if is_capa_salida:
                s = np.dot(derivadas, error)
                self.s.append(s)
            else:
                pesos = np.delete(self.pesos[i], 0, 1)
                matriz_jacobiana = np.dot(derivadas, pesos.T)
                s = np.dot(matriz_jacobiana, self.s[0])
                self.s = [s] + self.s


    def gradiente_descendente(self, rango_aprendizaje):
        for i in range(len(self.pesos)):
            nuevo_w = self.pesos[i] + rango_aprendizaje * np.dot(self.s[i], self.a[i].T)
            self.pesos[i] = nuevo_w
        

    def gradiente_descendente_lote(self, rango_aprendizaje):
        for i in range(len(self.pesos)):
            nuevo_w = self.pesos[i] + rango_aprendizaje * np.dot(self.s[i], self.a[i].T)            
            self.lote[i]+=nuevo_w
        self.cont+=1


    def fit(self, entradas, salidas_deseadas, epocas, rango_aprendizaje, error_deseado=None, plotter=None):
        convergido = False
        error_acumulativo = error_deseado if error_deseado else 1
        epoca_inicial = plotter.epoca_actual + 1
        epoca_final = epoca_inicial + epocas
        for epoca in range(epoca_inicial, epoca_final):
            if plotter:
                plotter.epoca_actual = epoca
            if error_deseado and error_acumulativo < error_deseado:
                convergido = True
                break
            error_acumulativo = 0
            for _entrada, salida_deseada in zip(entradas, salidas_deseadas):
                salida = self.forward_propagate(_entrada)
                salida_deseada = salida_deseada.reshape(salida.shape)
                error = salida_deseada - salida
                error_cuadrado = np.dot(error.T, error)
                self.back_propagate(error)
                error_acumulativo += error_cuadrado[0][0]                
                self.gradiente_descendente(rango_aprendizaje)
                
            if plotter:
                plotter.graficar_errores(error_acumulativo)
            # print( f"Error por epoca {epoca}: {error_acumulativo}")
        return convergido
        
        
    def quickprop(self, rango_aprendizaje):
        miu = 1.75  # Recommended value by Scott Fahlman
        es_primera_iteración_QP = self.ultimo_gradiente is None
        if es_primera_iteración_QP:
            self.ultimo_gradiente = []
            self.ultima_nabla_w = []
        for i in range(len(self.pesos)):
            gradiente = np.dot(self.s[i], self.a[i].T)
            if es_primera_iteración_QP:  
                nabla_w = rango_aprendizaje * gradiente
                self.ultimo_gradiente.append(deepcopy(gradiente))
                self.ultima_nabla_w.append(deepcopy(nabla_w))
            else:
                divisor = self.ultimo_gradiente[i] - gradiente
                divisor[divisor == 0] = 0.0001
                delta = gradiente / divisor
                temp = delta * self.ultima_nabla_w[i]
                factor_de_crecimiento_máximo = miu * self.ultima_nabla_w[i]

                temp_is_mayor_que_miu = temp > factor_de_crecimiento_máximo
                temp[temp_is_mayor_que_miu] = factor_de_crecimiento_máximo[temp_is_mayor_que_miu]
                ultimo_gradiente_y_producto_de_gradiente_actual = self.ultimo_gradiente[i] * gradiente
                gradiente_descendente = rango_aprendizaje * gradiente
                nabla_w = temp + gradiente_descendente

                gradientes_producto_es_menor_de_cero = ultimo_gradiente_y_producto_de_gradiente_actual < np.zeros(
                    ultimo_gradiente_y_producto_de_gradiente_actual.shape
                )
                nabla_w[gradientes_producto_es_menor_de_cero] = temp[gradientes_producto_es_menor_de_cero]
                nabla_demasiado_pequenio = nabla_w < np.ones(nabla_w.shape) * 1e-9
                nabla_w[nabla_demasiado_pequenio] = gradiente_descendente[nabla_demasiado_pequenio]
                self.ultimo_gradiente[i] = deepcopy(gradiente)
                self.ultima_nabla_w[i] = deepcopy(nabla_w)
            nuevo_w = self.pesos[i] + nabla_w
            self.pesos[i] = nuevo_w


    def fit_quick(self, entradas, salidas_deseadas, epocas, rango_aprendizaje, error_deseado=None, plotter=None):
        convergido = False
        error_acumulativo = error_deseado if error_deseado else 1
        epoca_inicial = plotter.epoca_actual + 1
        epoca_final = epoca_inicial + epocas
        error_alcanzado = [0, 0]
        for epoca in range(epoca_inicial, epoca_final):
            if plotter:
                plotter.epoca_actual = epoca
            if error_deseado and error_acumulativo < error_deseado:
                convergido = True
                break
            error_acumulativo = 0
            for _entrada, salida_deseada in zip(entradas, salidas_deseadas):
                salida = self.forward_propagate(_entrada)
                salida_deseada = salida_deseada.reshape(salida.shape)
                error = salida_deseada - salida
                error_cuadrado = np.dot(error.T, error)
                self.back_propagate(error)
                error_acumulativo += error_cuadrado[0][0]                
                self.quickprop(rango_aprendizaje)
                error_alcanzado[0] = error_acumulativo
                error_alcanzado[1] = epoca
                
            if plotter:
                plotter.graficar_errores(error_acumulativo)
            # print( f"Error por epoca {epoca}: {error_acumulativo}")
        self.error_alcanzado_QP=error_alcanzado
        return convergido


    def fit_lotes(self, entradas, salidas_deseadas, epocas, rango_aprendizaje, error_deseado=None, plotter=None):
        print("Por lotes...")
        convergido = False
        
        error_acumulativo = error_deseado if error_deseado else 1
        epoca_inicial = plotter.epoca_actual + 1
        epoca_final = epoca_inicial + epocas
        
        for epoca in range(epoca_inicial, epoca_final):
            self.lote=deepcopy(self.pesos)
            self.cont=1
            if plotter:
                plotter.epoca_actual = epoca
            if error_deseado and error_acumulativo < error_deseado:
                convergido = True
                break
            error_acumulativo = 0
            for _entrada, salida_deseada in zip(entradas, salidas_deseadas):
                salida = self.forward_propagate(_entrada)
                salida_deseada = salida_deseada.reshape(salida.shape)
                error = salida_deseada - salida
                error_cuadrado = np.dot(error.T, error)
                self.back_propagate(error)
                error_acumulativo += error_cuadrado[0][0]                
                self.gradiente_descendente_lote(rango_aprendizaje)
            self.promedio_y_actualiza()
            if plotter:
                plotter.graficar_errores(error_acumulativo)
            # print( f"Error por epoca {epoca}: {error_acumulativo}")
        return convergido
    
    
    def promedio_y_actualiza(self):#calula el promedido de los gradientes y actualiza pesos
        cont=self.cont
       
        for a in self.lote:
            for i in range(len(a)):
                a[i]=a[i]/cont
        for i in range(len(self.lote)):       
            self.pesos[i] = self.lote[i]
        #print("cont:"+str(self.cont))


    def guess(self, _entrada):
        salida = self.forward_propagate(_entrada)
        salida = np.array([x for x in salida])
        return salida


    def _funcion_de_activacion(self, x):  
        return 1 / (1 + np.exp(-x))


    def _funcion_de_activacion_derivada(self, x):
        return x * (1.0 - x)

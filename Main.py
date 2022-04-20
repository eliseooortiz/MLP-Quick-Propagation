from cProfile import label
from msilib.schema import RadioButton
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons
import matplotlib as mpl
from multilayerperceptron import MultiLayerPerceptron

class Ventana:
    puntos, clase_deseada = np.array([]), []
    mlp=None
    epoca_actual=0
    epocas_maximas=0
    rango=0.1
    texto_de_epoca = None
    texto_de_convergencia= None
    error_minimo=0.1
    neuronas_capa=None
    rango_inicializado=False
    pesos_inicializados=False
    mlp_entrenado=False
    lineas=[]
    termino=False
    errores=[]
    puntos_barrido=[]
    clase=0
    colores = ['b', 'r', 'g', 'm', 'c', 'y']
    marcadores = ['x', '.', '^', 's']
    comobinacion_marcadores=[color + marker for marker, color in zip(marcadores, colores)]
    marcadores_de_linea=[c + '-' for c in reversed(colores)]
    tipo_gradiente=0
    quick_entrenado=False


    def __init__(self):
        #Configuracion inicial de la interfaz grafica.
        mpl.rcParams['toolbar'] = 'None'
        self.fig, (self.grafica, self.grafica_errores) = plt.subplots(1, 2)
        self.fig.canvas.manager.set_window_title('Multilayer Perceptron - MLP')
        self.fig.set_size_inches(13, 7, forward=True)
        plt.subplots_adjust(bottom=0.220, top=0.920)
        self.grafica.set_xlim(-5.0,5.0)
        self.grafica.set_ylim(-5.0,5.0)
        self.fig.suptitle("Algoritmo MLP")
        self.grafica_errores.set_title("Errores")
        self.grafica_errores.set_xlabel("Época")
        self.grafica_errores.set_ylabel("Error")
        # Acomodo de los botones y cajas de texto
        cordenadas_rango = plt.axes([0.210, 0.10, 0.07, 0.03])
        coordenadas_epcoas = plt.axes([0.420, 0.10, 0.07, 0.03])
        coordenadas_error_deseado = plt.axes([0.210, 0.05, 0.07, 0.03])
        coordenadas_capas=plt.axes([0.420, 0.05, 0.07, 0.03])
        coordenadas_pesos = plt.axes([0.510, 0.05, 0.125, 0.03])
        coordenadas_clase = plt.axes([0.510, 0.10, 0.125, 0.03])
        coordenadas_quick = plt.axes([0.650, 0.05, 0.1, 0.03])
        coordenadas_entrenar_mlp= plt.axes([0.650, 0.10, 0.1, 0.03])
        
        self.text_box_rango = TextBox(cordenadas_rango, "Rango de aprendizaje: ")
        self.text_box_epocas = TextBox(coordenadas_epcoas, "Épocas maximas: ")
        self.text_box_error_minimo_deseado = TextBox(coordenadas_error_deseado, "Error mínimo deseado: ")
        self.text_box_capas = TextBox(coordenadas_capas, "Neuronas por capa (,): ")
        boton_pesos = Button(coordenadas_pesos, "Inicializar pesos")
        self.boton_clase = Button(coordenadas_clase, "Clase 0,1")
        boton_quick = Button(coordenadas_quick, "Quick")
        boton_entrenar_mlp = Button(coordenadas_entrenar_mlp, "BP")
        
        self.text_box_epocas.on_submit(self.validar_epocas)
        self.text_box_rango.on_submit(self.validar_rango)
        self.text_box_error_minimo_deseado.on_submit(self.validar_error_minimo_deseado)
        self.text_box_capas.on_submit(self.validar_capas)
        boton_pesos.on_clicked(self.inicializar_pesos)
        self.boton_clase.on_clicked(self.cambio_clase)
        boton_quick.on_clicked(self.entrenar_quick)
        boton_entrenar_mlp.on_clicked(self.entrenar_mlp)
        self.fig.canvas.mpl_connect('button_press_event', self.__onclick)
        plt.show()



    def __onclick(self, event):
        if event.inaxes == self.grafica:
            punto_actual= [event.xdata, event.ydata]    
            self.puntos = np.append(self.puntos, punto_actual).reshape([len(self.puntos) + 1, 2])
            is_left_click = event.button == 1  
            clase=  str(int(self.clase)) + str(int(not is_left_click)) #obtiene el string en binario de las clases  
            aux=int(clase,2)      #obtiene el valor del binario  
            self.clase_deseada.append([int(x) for x in clase])
            self.grafica.plot(event.xdata, event.ydata, self.comobinacion_marcadores[aux])
            self.fig.canvas.draw()


    
    def graficar_errores(self,cumulative_error):   
        self.errores.append(cumulative_error)
        x=self.errores
        y = range(len(x))
        self.grafica_errores.clear()
        self.grafica_errores.plot(y,x)
        plt.pause(0.3)


    def cambio_clase(self,event):
        self.clase= not self.clase
        if(self.clase):
            self.boton_clase.label.set_text("clase 2,3")
        else:    
            self.boton_clase.label.set_text("clase 0,1")


    def inicializar_pesos(self, event):
        if self.rango_inicializado and self.epocas_maximas>0 and len(self.puntos)>0 and not self.mlp_entrenado and self.neuronas_capa is not None:
            self.mlp = MultiLayerPerceptron(
                2,
                self.neuronas_capa,
                2,
                [-5.0, 5.0]
            )
            self.mlp.init_weights()
            self.graficar_lineas(self.mlp.weights[0])
            self.pesos_inicializados = True
        

    def graficar_lineas(self,pesos):
        aux=len(self.lineas)==0
        if(aux):
            self.texto_de_epoca = self.grafica.text(3,5, 'Epoca: %s' % self.epoca_actual,fontsize=10)
        else:
            self.texto_de_epoca.set_text('Epoca: %s' % self.epoca_actual)
        x1 = np.array([self.puntos[:, 0].min() - 10, self.puntos[:, 0].max() + 10])
        for i in range(len(pesos)):
            peso=pesos[i]
            m = -peso[1] / peso[2]
            c = peso[0] / peso[2]
            x2 = m * x1 + c
            if(aux):
                linea, = self.grafica.plot(
                    x1,
                    x2,
                    self.marcadores_de_linea[i % len(self.marcadores_de_linea)]
                )
                self.lineas.append(linea)
            else:
                self.lineas[i].set_xdata(x1)
                self.lineas[i].set_ydata(x2)
            self.fig.canvas.draw()
            plt.pause(0.1)

    
    def validar_rango(self, expression):
        try:
            r=float(expression)
            if(r>0 and r<1):
                self.rango =float(expression)
            else:
                self.rango =0.1    
        except ValueError:
            self.rango =0.1
        finally:
            self.text_box_rango.set_val(self.rango)
            self.rango_inicializado=True


    def validar_epocas(self, expression):
        try:
            self.epocas_maximas =int(expression)
        except ValueError:
            self.epocas_maximas = 50
        finally:
            self.text_box_epocas.set_val(self.epocas_maximas)


    def validar_error_minimo_deseado(self,expression):
        try:
            r=float(expression)
            if(r>0 and r<1):
                self.error_minimo =float(expression)
            else:
                self.error_minimo =0.1    
        except ValueError:
            self.error_minimo =0.1
        finally:
            self.text_box_error_minimo_deseado.set_val(self.error_minimo)
    
    
    def validar_capas(self,expression):
        value = 0
        try:
            value = eval(expression)
        except (SyntaxError, NameError):
            if expression:
                value = 0
                self.text_box_capas.set_val(value)

        if type(value) == tuple:
            self.neuronas_capa = [x for x in value]
        else:
            self.neuronas_capa = [value]


    

    def barrido(self):
        y = 5
        while(y>=-5):
            x=-5
            while x<=5:
                p=[x,y]
                g=self.mlp.guess(p)
                clase=[g[0][0],g[1][0]]
                if(clase[0]<.5):#clase 0 y 1
                    if(clase[1]>.5):#clase 1
                        alpha=clase[1]
                        alpha = (alpha - .5) * 2 
                        punto,=self.grafica.plot(x,y, color=(1,0,0, alpha), marker='.')#rojo
                        self.puntos_barrido.append(punto)
                    else:#clase 0
                        alpha=clase[0]
                        #if(clase[0]>.5):
                        #    alpha = (alpha -.5) * 2 
                        #else:
                        #    alpha = (alpha ) * 2 
                        punto,=self.grafica.plot(x,y, color=(0,0,1, 1), marker='.')#azul
                        self.puntos_barrido.append(punto)
                else:#clase 2 y 3
                    if(clase[1]>.5):#clase 4
                        alpha=clase[1]
                        alpha = (alpha - .5) * 2 
                        punto,=self.grafica.plot(x,y, color=(1,0,1, alpha), marker='.')#magenta
                        self.puntos_barrido.append(punto)
                    else:#clase 3
                        alpha=clase[0]
                        if(clase[0]>.5):
                            alpha = (alpha -.5) * 2 
                        else:
                            alpha = (alpha ) * 2                         
                        punto,=self.grafica.plot(x,y, color=(0,1,0, alpha), marker='.')#verde 
                        self.puntos_barrido.append(punto)
                x+=.05
            y-=.05
            print(y)
        self.grafica.set_xlim(-5.0,5.0)
        self.grafica.set_ylim(-5.0,5.0)
        """
            colores = ['b', 'r', 'g', 'm', 'c', 'y']
            marcadores = ['x', '.', '^', 's']
        """
        for j,k in enumerate(self.puntos):
            if(self.clase_deseada[j][0]==0):
                if(self.clase_deseada[j][1]==0):
                    self.grafica.plot(k[0], k[1],'rx')
                else:
                    self.grafica.plot(k[0], k[1],'b.')
            else:
                if(self.clase_deseada[j][1]==0):
                    self.grafica.plot(k[0], k[1],'m^')
                else:
                    self.grafica.plot(k[0], k[1],'gs')
        self.fig.canvas.draw()

    def limpiar_barrido(self):
        self.grafica.set_xlim(-5.0,5.0)
        self.grafica.set_ylim(-5.0,5.0)
        """
            colores = ['b', 'r', 'g', 'm', 'c', 'y']
            marcadores = ['x', '.', '^', 's']
        """
        for linea in self.lineas:
            linea.remove()
        self.lineas = []
        for punto in self.puntos_barrido:
            punto.remove()
        self.puntos_barrido = []
        for j,k in enumerate(self.puntos):
            if(self.clase_deseada[j][0]==0):
                if(self.clase_deseada[j][1]==0):
                    self.grafica.plot(k[0], k[1],'bx')
                else:
                    self.grafica.plot(k[0], k[1],'r.')
            else:
                if(self.clase_deseada[j][1]==0):
                    self.grafica.plot(k[0], k[1],'g^')
                else:
                    self.grafica.plot(k[0], k[1],'ms')
        self.fig.canvas.draw()
    def entrenar_mlp(self, event):
        self.limpiar_barrido()
        learning_rate_initialized = self.rango != 0
        max_epochs_initialized = self.epocas_maximas != 0
        desired_error_is_set = self.error_minimo != 0.0
        hyper_params_are_set = learning_rate_initialized == max_epochs_initialized == desired_error_is_set is True
        if not self.mlp_entrenado and self.pesos_inicializados and hyper_params_are_set:
            self.clase_deseada = np.array(self.clase_deseada)
            if(self.tipo_gradiente==0):
                converged = self.mlp.fit(self.puntos, self.clase_deseada, self.epocas_maximas, self.rango, self.error_minimo, self)
            else:
                converged = self.mlp.fit_lotes(self.puntos, self.clase_deseada, self.epocas_maximas, self.rango, self.error_minimo, self)
            convergence_text = "convergio" if converged else "no convergio"
            if self.texto_de_convergencia:
                self.texto_de_convergencia.set_text(convergence_text)
            else:
                self.texto_de_convergencia = self.grafica.text(
                    -0.25,
                    0.9,
                    convergence_text,
                    fontsize=10
                )
            if self.quick_entrenado:
                self.texto_de_epoca.set_text("Epoca: %s \n Epoca Qp: %s  Error Qp: %s" % (self.epoca_actual ,self.mlp.error_reached_QP[1],self.mlp.error_reached_QP[0]))    
            else:
                self.texto_de_epoca.set_text("Epoca: %s" % self.epoca_actual)
            plt.pause(0.1)
        self.barrido()
        plt.pause(0.1)

    def entrenar_quick(self, event):
        self.limpiar_barrido()
        learning_rate_initialized = self.rango != 0
        max_epochs_initialized = self.epocas_maximas != 0
        desired_error_is_set = self.error_minimo != 0.0
        hyper_params_are_set = learning_rate_initialized == max_epochs_initialized == desired_error_is_set is True
        if not self.mlp_entrenado and self.pesos_inicializados and hyper_params_are_set:
            self.clase_deseada = np.array(self.clase_deseada)
            
            converged = self.mlp.fit_quick(self.puntos, self.clase_deseada, self.epocas_maximas, self.rango, self.error_minimo, self)
            self.quick_entrenado=True
            convergence_text = "convergio" if converged else "no convergio"
            if self.texto_de_convergencia:
                self.texto_de_convergencia.set_text(convergence_text)
            else:
                self.texto_de_convergencia = self.grafica.text(
                    -0.25,
                    0.9,
                    convergence_text,
                    fontsize=10
                )
            self.texto_de_epoca.set_text("Epoca: %s" % self.epoca_actual)
            plt.pause(0.1)
        self.barrido()
        plt.pause(0.1)
    


if __name__ == '__main__':
    Ventana()
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import NewsCore.model.News as newsModel
from cassandra import ReadTimeout
import NewsCore.dao.NewsDAOImpl as dao

if __name__ == '__main__':
    example1 = dao.NewsDAOImpl()#'mydb'
    example1.createsession('localhost')
    example1.setlogger()
    example1.loadkeyspace('mydb')
    example1.create_table()

    n4_energy = newsModel.News(8, "energy", 10, "8PUE de Centro de Datos 1 anormalmente alto",
                               "ayor que en otros centros de datos y un 70% mayor que el estado del arte. Este problema supone un sobrecoste anual de 0,3 M€, un 20% del coste total de electricidad para Centro de Datos 1.",
                               "", "", ["pue"])
    example1.insert_data(n4_energy)
    # n1_energy = newsModel.News(1, "energy", 12, "12PUE de Centro de Datos 1 anormalmente alto",
    #                            "El valor del PUE (Power Usage Effectiveness) es un 50% mayor que en otros centros de datos y un 70% mayor que el estado del arte. Este problema supone un sobrecoste anual de 0,3 M€, un 20% del coste total de electricidad para Centro de Datos 1.",
    #                            "", "", ["pue", "coste energía refrigeración", "coste energía total"])
    # n2_energy = newsModel.News(1, "energy", 12, "12HOLAA",
    #                            "El valor del PUE (Power Usage Effectiveness) es un 50% mayor que en otros centros de datos y un 70% mayor que el estado del arte. Este problema supone un sobrecoste anual de 0,3 M€, un 20% del coste total de electricidad para Centro de Datos 1.",
    #                            "", "", ["pue"])
    # n3_energy = newsModel.News(1, "energy", 15, "15PUE de Centro de Datos 1 anormalmente alto",
    #                            "El valor del PUE (Power Usage Effectiveness) es un 50% mayor que en otros centros de datos y un 70% mayor que el estado del arte. Este problema supone un sobrecoste anual de 0,3 M€, un 20% del coste total de electricidad para Centro de Datos 1.",
    #                            "", "", ["pue", "coste energía refrigeración", "coste energía total"])
    # example1.insert_data(n1_energy)
    # example1.insert_data(n2_energy)
    # example1.insert_data(n3_energy)



    # n2_energy = newsModel.News("", "", 2, "energy", 8,
    #                     "La baja ocupación IT de Centro de Datos 1 sugiere actuar para reducir el desperdicio de energía",
    #                     [
    #                         "El nivel de ocupación IT es un 50% menor que en otros centros de datos y se ha mantenido bajo en los últimos 15 meses.",
    #                         "Centro de Datos 1 esta diseñado para un consumo de potencia IT de 1 MW y actualmente se consume sólo el 15%."],
    #                     "", "", ["pue", "uso de capacidad de potencia de it", "ocupacion it"])
    #
    # n3_energy = newsModel.News("", "", 3, "energy", 9,
    #                     "La energía de refrigeración en Centro de Datos 1 es similar a la energia de IT",
    #                     ["Centro de Datos 1 consume anualmente 0,5 GWh sólo en la refrigeración de los equipos IT.",
    #                      "El consumo de energía de IT es tan bajo, que la refrigeración tiene un peso importante en el PUE.",
    #                      "El PUE actual está por encima del valor medio mundial."],
    #                     "", "", ["pue", "energía refrigeración", "energía it"])
    #
    # n4_energy = newsModel.News("", "", 4, "energy", 6, "Los picos de consumo de Centro de Datos 1 no son significativos", [
    #     "La potencia de refrigeración se mantiene con variaciones poco significativas entre 68 kW y 77 kW.",
    #     "La variación de la potencia de IT es alta, pero la potencia máxima se queda por debajo del 15% de la capacidad de potencia de la infraestructura."],
    #                     "", "", ["pue", "potencia máxima refrigeración", "potencia máxima it"])
    #
    # n1_environment = newsModel.News("", "", 1, "environment", 12, "CUE de Centro de Datos 1 anormalmente alto", [
    #     "El valor del CUE (Carbon Usage Effectiveness) en Centro de Datos 1 es 53% mayor que en otros centros de datos, y un 70% mayor que el estado del arte.",
    #     "Centro de Datos 1 opera con un PUE (Power Usage Effectiveness) de 1,98, desperdiciando mas de 0,3 GWh anuales de electricidad."],
    #                          "", "", ["pue", "cue", "coste energía refrigeración"])
    #
    # n2_environment = newsModel.News("", "", 2, "environment", 7, "Emisiones de CO2 de Centro de Datos 1 anormalmente altas", [
    #     "Centro de Datos 1 es responsable del 13% de las emisiones de CO2 y la mitad podrían evitarse."],
    #                          "", "", ["emisiones refrigeración", "ocupación it", "uso de capacidad de potencia de IT"])
    #
    # n3_environment = newsModel.News("", "", 3, "environment", 6, "Energía en Centro de Datos 1 anormalmente alta", [
    #     "El valor de la energía en Centro de Datos 1 es 53% mayor que en otros centros de datos, y un 70% mayor que el estado del arte."],
    #                          "", "", ["energia refrigeración", "energía it"])
    #
    # n1_financial = newsModel.News("", "", 1, "financial", 8, "Centro de Datos 1 desperdicia 0,3 M€ al año en refrigeración", [
    #     "El valor de la energía en Centro de Datos 1 es 53% mayor que en otros centros de datos, y un 70% mayor que el estado del arte."],
    #                        "", "", ["coste energía refrigeración"])
    #
    # n2_financial = newsModel.News("", "", 2, "financial", 4,
    #                        "La baja ocupación IT de Centro de Datos 1 sugiere actuar para reducir el coste de la infraestructura de refrigeración",
    #                        [
    #                            "El nivel de ocupacion IT es un 50% menor que en otros centros de datos y se ha mantenido bajo en los ultimos 15 meses.",
    #                            "Centro de Datos 1 esta disenado para un consumo de potencia IT de 1 MW y actualmente se consume solo el 15%."],
    #                        "", "", ["energia refrigeracion", "energia it"])
    #
    # n3_financial = newsModel.News("", "", 3, "financial", 13, "Distribución de costes de energía en Centro de Datos 1", [
    #     "El coste de la energía de refrigeración supone el 49,5% del coste de energía total de Centro de Datos 1"],
    #                        "", "",
    #                        ["coste energía refrigeración", "coste energía it", "energía it", "energía refrigeración"])
    #
    # n4_financial = newsModel.News("", "", 4, "financial", 2, "Extremados niveles en Centro de Datos 1 de CUE, WUE y PUE",
    #                        ["Bajo nivel de DCE"],
    #                        "", "", ["pue", "wue", "dce", "cue"])
    #
    # news = [n1_energy, n2_energy, n3_energy, n4_energy, n1_environment, n2_environment, n3_environment, n1_financial,
    #         n2_financial, n3_financial, n4_financial]
    #
    # for i in news:
    #     example1.insert_data(i)


    #example1.select_data(3)
    # AAdata = example1.selectNews(keywords=['coste energía refrigeración', 'pue'])
    # print(AAdata)
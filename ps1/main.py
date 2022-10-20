import ps1.exercises.zadanie1 as z1
import ps1.exercises.zadanie2 as z2
import ps1.exercises.zadanie3 as z3


def run(number: int):
    zadania = {
        '1': z1.run,
        '2': z2.run,
        '3': z3.run,
    }
    zadania[number]()
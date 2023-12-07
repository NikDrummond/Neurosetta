# Module for iterative screening of neurons

import vedo as vd
import numpy as np

## Plot function

def Check_plot(meshes):

    # button behaviour functions
    def scroll_left(obj, ename):
        global index
        global flags
        i = (index - 1) % len(meshes)
        if flags[i]:
            txt.text(str(i) +  ': ' + meshes[i].name + ' - Flag').c("r")
        else:
            txt.text(str(i) +  ': ' + meshes[i].name + ' - No Flag').c("k")
        plt.remove(meshes[index]).add(meshes[i])
        plt.reset_camera()
        index = i

    def scroll_right(obj, ename):
        global index
        global flags
        i = (index + 1) % len(meshes)
        if flags[i]:
            txt.text(str(i) +  ': ' + meshes[i].name + ' - Flag').c("r")
        else:
            txt.text(str(i) +  ': ' + meshes[i].name + ' - No Flag').c("k")
        plt.remove(meshes[index]).add(meshes[i])
        plt.reset_camera()
        index = i

    def flag(obj, ename):
        global flags
        global index
        # check current state (flagged or not)
        # if currently flagged:
        if flags[index]:
            # update text
            txt.text(str(index) +  ': ' + meshes[index].name + ' - No Flag').c("k")
            # update state
            flags[index] = False
        # if not currently flagged
        else:
            # update text
            txt.text(str(index) +  ': ' + meshes[index].name + ' - Flag').c("r")
            # update state
            flags[index] = True
        # update plot
        plt.reset_camera()


    # initiate flags and index
    flags = np.zeros_like(meshes).astype(bool)
    index = 0
    # text representing plot title
    if flags[index]:
        title_1 = str(index) + ': ' + meshes[0].name + ' - Flag'
        txt = vd.Text2D(title_1, font="Courier", pos="top-center", s=1.5, c = 'r')
    else:
        title_1 = str(index) + ': ' + meshes[0].name + ' - No Flag'
        txt = vd.Text2D(title_1, font="Courier", pos="top-center", s=1.5, c = 'k')

    # plotter
    plt = vd.Plotter()
    # buttons
    bu = plt.add_button(
        scroll_right,
        pos=(0.8, 0.06),  # x,y fraction from bottom left corner
        states=[">"],     # text for each state
        c=["w"],          # font color for each state
        bc=["k5"],        # background color for each state
        size=40,          # font size
    )
    bu = plt.add_button(
        scroll_left,
        pos=(0.2, 0.06),  # x,y fraction from bottom left corner
        states=["<"],     # text for each state
        c=["w"],          # font color for each state
        bc=["k5"],        # background color for each state
        size=40,          # font size
    )
    bu = plt.add_button(
        flag,
        pos=(0.5, 0.06),
        states=["Flag"],
        c=["w"],
        bc=["r"],
        size=40,
    )
    # generate plot
    plt += txt
    plt.show(meshes[0]).close()




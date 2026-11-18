import type GUI from 'lil-gui';

export function help(gui: GUI, message: string) {
    const paragraph = document.createElement('p');
    paragraph.className = 'control-help';
    paragraph.textContent = message;
    gui.$children.append(paragraph);
    return paragraph;
}

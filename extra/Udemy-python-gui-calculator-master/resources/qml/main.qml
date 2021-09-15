import QtQuick 2.10
import QtQuick.Controls 2.4
import QtQuick.Layouts 1.4
import "customs"

ApplicationWindow {
    visible: true
    width: 320
    height: 580

    property string screenProblem: "0"
    property string bg_problem: ""
    property string evaluation: ""
    property color func_color: "#f1f1f1"
    property color mem_color: "#75000000"
    property bool computed: false
    property bool portrait: width < height


    signal btn_click(string no)

    signal mem_click(string mem_str)

    signal uniClick(string code, string repl)

    signal deleteText()

    onBtn_click: {
        var stat;
        stat = no



        if(computed) {
            screenProblem = "0"
            bg_problem = "0"
            evaluation = ""
            computed = false
        }

        if(screenProblem != "0") {
            screenProblem += stat
            bg_problem += stat

        } else if(no == '.' && screenProblem == '0') {
            screenProblem = '0.';
            bg_problem = '0.';

        } else {
            screenProblem = stat
            bg_problem = stat;

        }
        if(scr_lab.width > answer_sheet.width) {
            flicker.flick(-512, 0)
        }

    }

    onUniClick: {

        if(computed) {
            screenProblem = "0"
            bg_problem = "0"
            evaluation = ""
            computed = false
        }

        var stat = "<span style='font-family: Segoe MDL2 Assets; font-size: 18px;' > " + code + " </span>"
        if(screenProblem != "0") {
            screenProblem += stat
            bg_problem += repl
        } else {
            screenProblem = stat
            bg_problem = repl
        }
    }

    onDeleteText: {
        let len = screenProblem.length - 1
        let new_string = screenProblem.substring(0, len);
        if (new_string == '' ) {
            new_string = '0'
        }
        screenProblem = new_string;
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {// screen
            id: answer_sheet
            Layout.alignment: Qt.AlignCenter
            Layout.fillWidth: true
            Layout.preferredHeight: width > 300 ? 128 : 48
            color: "white"

            ColumnLayout {
                width: parent.width
                anchors.right: parent.right
                anchors.rightMargin: 8
                anchors.verticalCenter: parent.verticalCenter

                Rectangle {
                    id: problem_screen
                    Layout.alignment: Qt.AlignRight
                    Layout.preferredWidth: scr_lab.width
                    Layout.maximumWidth: parent.parent.width
                    Layout.preferredHeight: 48

                    Flickable {
                        id: flicker
                        width: parent.width
                        height: parent.height
                        flickableDirection: Flickable.HorizontalFlick
                        contentWidth: scr_lab.width
                        contentHeight: scr_lab.height
                        contentX: 0
                        contentY: 0

                        Label {
                            id: scr_lab
                            leftPadding: 8
                            text: screenProblem
                            textFormat: Text.RichText
                            elide: Text.ElideMiddle
                            font.family: "Segoe UI Semilight"
                            font.pixelSize: 42
                            font.bold: true
                        }

                    }
                }

                Rectangle {
                    Layout.alignment: Qt.AlignRight
                    Layout.preferredWidth: eval_lab.width
                    Layout.maximumWidth: parent.parent.width
                    Layout.preferredHeight: 48

                    Flickable {
                        width: parent.width
                        height: parent.height
                        flickableDirection: Flickable.HorizontalFlick
                        contentWidth: eval_lab.width
                        contentHeight: eval_lab.height
                        contentX: 0
                        contentY: 0

                        Label {
                            id: eval_lab
                            leftPadding: 8
                            Layout.alignment: Qt.AlignRight
                            text: evaluation
                            font.family: "Segoe UI Semilight"
                            font.pixelSize: 24
                            font.bold: true
                            color: "dodgerblue"
                        }

                    }
                }

            }

            Rectangle { // border
                anchors.bottom: parent.bottom
                width: parent.width
                height: 1
                color: "#ebebeb"
            }

        }

        Rectangle {// buttons
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "transparent"

            Rectangle {// s size
                anchors.fill: parent
                visible: portrait
                color: "#ebebeb"

                GridLayout {
                    anchors.fill: parent
                    columns: 4
                    rowSpacing: 1
                    columnSpacing: 1

                    CalcButton {
                        text: "mc"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: mem_color
                        bg_color: func_color

                        onClicked: mem_click(this.text)

                    }

                    CalcButton {
                        text: "m+"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: mem_color
                        bg_color: func_color

                        onClicked: mem_click(this.text)

                    }

                    CalcButton {
                        text: "m-"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: mem_color
                        bg_color: func_color

                        onClicked: mem_click(this.text)

                    }

                    CalcButton {
                        text: "mr"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: mem_color
                        bg_color: func_color

                        onClicked: mem_click(this.text)

                    }

                    CalcButton {
                        text: "C"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: "dodgerblue"
                        bg_color: func_color

                        onClicked: screenProblem = "0"

                    }

                    CalcButton {
                        text: "\uE94A"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: "dodgerblue"
                        bg_color: func_color

                        onClicked: uniClick(this.text, '/')

                    }

                    CalcButton {
                        text: "\uE947"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: "dodgerblue"
                        bg_color: func_color

                        onClicked: uniClick(this.text, '*')

                    }

                    CalcButton {
                        text: "\uE94F"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: "dodgerblue"
                        bg_color: func_color

                        onClicked: deleteText()

                    }

                    CalcButton {
                        text: "7"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "8"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "9"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "\uE949"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: "dodgerblue"
                        bg_color: func_color

                        onClicked: uniClick(this.text, '-')

                    }

                    CalcButton {
                        text: "4"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "5"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "6"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "\uE948"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1
                        txt_color: "dodgerblue"
                        bg_color: func_color

                        onClicked: uniClick(this.text, '+')

                    }

                    CalcButton {
                        text: "1"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "2"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "3"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "\uE94E"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 2
                        Layout.columnSpan: 1
                        txt_color: "white"
                        bg_color: "dodgerblue"

                        onClicked: {
                            computed = true
                            Calculator.compute(bg_problem)
                        }

                    }

                    CalcButton {
                        text: "\uE94C"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: uniClick(this.text, '%')

                    }

                    CalcButton {
                        text: "0"
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                    CalcButton {
                        text: "."
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.rowSpan: 1
                        Layout.columnSpan: 1

                        onClicked: btn_click(this.text)

                    }

                }

            }

            Rectangle {// fullsize
                anchors.fill: parent
                color: "transparent"
                visible: !portrait

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: parent.width > 1000 ? 48 : 24
                    anchors.rightMargin: parent.width > 1000 ? 48 : 24
                    anchors.topMargin: parent.width > 100 ? 54 : 24
                    anchors.bottomMargin: parent.width > 100 ? 54 : 24
                    spacing: parent.width > 600 ? 48 : 12

                    Rectangle {// functions
                        Layout.fillWidth: true
                        Layout.maximumWidth: 620
                        Layout.fillHeight: true
                        Layout.minimumHeight: 320
                        color: "lightgrey"

                        GridLayout {
                            anchors.fill: parent
                            anchors.margins: 1
                            columns: 4
                            rowSpacing: 1
                            columnSpacing: 1

                            CalcButton {
                                text: "mc"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: mem_color
                                bg_color: func_color

                                onClicked: mem_click(this.text)

                            }

                            CalcButton {
                                text: "m+"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: mem_color
                                bg_color: func_color

                                onClicked: mem_click(this.text)

                            }

                            CalcButton {
                                text: "m-"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: mem_color
                                bg_color: func_color

                                onClicked: mem_click(this.text)

                            }

                            CalcButton {
                                text: "mr"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: mem_color
                                bg_color: func_color

                                onClicked: mem_click(this.text)

                            }

                            CalcButton {
                                text: "C"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: "dodgerblue"
                                bg_color: func_color

                                onClicked: screenProblem = "0"

                            }

                            CalcButton {
                                text: "\uE94A"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: "dodgerblue"
                                bg_color: func_color

                                onClicked: uniClick(this.text, '/')

                            }

                            CalcButton {
                                text: "\uE947"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: "dodgerblue"
                                bg_color: func_color

                                onClicked: uniClick(this.text, '*')

                            }

                            CalcButton {
                                text: "\uE94E"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 2
                                Layout.columnSpan: 1
                                txt_color: "white"
                                bg_color: "dodgerblue"

                                onClicked: {
                                    computed = true
                                    Calculator.compute(bg_problem)
                                }

                            }

                            CalcButton {
                                text: "\uE94F"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: "dodgerblue"
                                bg_color: func_color

                                onClicked: deleteText()

                            }

                            CalcButton {
                                text: "\uE949"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: "dodgerblue"
                                bg_color: func_color

                                onClicked: uniClick(this.text, '-')

                            }

                            CalcButton {
                                text: "\uE948"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1
                                txt_color: "dodgerblue"
                                bg_color: func_color

                                onClicked: uniClick(this.text, '+')

                            }

                        }

                    }

                    Rectangle {// numbers
                        Layout.fillWidth: true
                        Layout.maximumWidth: 480
                        Layout.fillHeight: true
                        Layout.minimumHeight: 320
                        Layout.alignment: Qt.AlignRight
                        color: "lightgrey"

                        GridLayout {
                            anchors.fill: parent
                            anchors.margins: 1
                            columns: 3
                            rowSpacing: 1
                            columnSpacing: 1


                            CalcButton {
                                text: "7"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "8"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "9"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "4"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "5"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "6"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "1"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "2"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "3"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "\uE94C"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: uniClick(this.text, '%')

                            }

                            CalcButton {
                                text: "0"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                            CalcButton {
                                text: "."
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.rowSpan: 1
                                Layout.columnSpan: 1

                                onClicked: btn_click(this.text)

                            }

                        }

                    }


                }

            }

        }

    }

    Connections {
        target: Calculator

        onEvaluated: {
            var result = _compute
            evaluation = result
        }

    }

}

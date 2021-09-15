import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1364
    height: 700
    title: qsTr('Grid Layout')


    GridLayout {
        width: parent.width
        //height: parent.height
        rows: 3
        columns: 4
        rowSpacing: 2
        columnSpacing: 2

        Rectangle {
            Layout.fillWidth: true
            Layout.columnSpan: 3
            height: 360
            color: "dodgerblue"
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.columnSpan: 1
            height: 360
            color: "dodgerblue"
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.columnSpan: 2
            height: 240
            color: "transparent"

            Grid {
                width: parent.width
                height: parent.height
                rows: 3
                columns: 2
                rowSpacing: 2
                columnSpacing: 2

                Rectangle {
                    width: parent.width / 2 - 1
                    height: parent.height / 2 - 1
                    color: "#9BCC29"
                }

                Rectangle {
                    width: parent.width / 2 - 1
                    height: parent.height / 2 - 1
                    color: "#9BCC29"
                }

                Rectangle {
                    width: parent.width / 2 - 1
                    height: parent.height / 2 - 1
                    color: "#9BCC29"
                }

                Rectangle {
                    width: parent.width / 2 - 1
                    height: parent.height / 2 - 1
                    color: "#9BCC29"
                }

            }

        }

        Rectangle {
            Layout.fillWidth: true
            Layout.columnSpan: 1
            height: 240
            color: "dodgerblue"
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.columnSpan: 1
            height: 240
            color: "dodgerblue"
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.columnSpan: 1
            //height: 240
            //color: "dodgerblue"
        }

    }

}

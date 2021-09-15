import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3
import "addon"

ApplicationWindow {
    visible: true
    width: 1364
    height: 700
    title: qsTr('Organised Layout')


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

            CustomisedLayout {
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

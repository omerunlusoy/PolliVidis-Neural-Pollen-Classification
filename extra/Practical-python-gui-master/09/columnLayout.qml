import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 1280
    height: 700

    ColumnLayout {
        width: parent.width

        RowLayout {
            //Layout.fillWidth: true
            height: 300

            Rectangle {
                Layout.preferredWidth: 700
                Layout.fillHeight: true
                color: "#2993cc"
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                color: "#F7630C"
            }

        }

        Rectangle {
            Layout.fillWidth: true
            height: 300
            color: "lime"
        }

        Rectangle {
            Layout.fillWidth: true
            height: 300
            color: "orange"
        }

    }

}

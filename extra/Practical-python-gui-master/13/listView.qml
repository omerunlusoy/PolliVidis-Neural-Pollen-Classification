import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

ApplicationWindow {
    visible: true
    width: 512
    height: 512

    ListView {
        width: parent.width
        height: parent.height


        model: Users {}

        header: Rectangle {
            width: parent.width
            height: 44

            Text {
                padding: 12
                text: "Choose Account"
                font.pixelSize: 20
                color: "dodgerblue"
            }

        }

        delegate: Rectangle {
                width: parent.width
                height: 68
                color: "transparent"

                RowLayout {
                    height: parent.height
                    spacing: 16

                    Image {
                        anchors {
                            verticalCenter: parent.verticalCenter
                            left: parent.left
                            leftMargin: 12
                        }

                        sourceSize.width: 64
                        sourceSize.height: 64
                        source: profile
                    }

                    ColumnLayout {
                        width: parent.parent.width

                        Text {
                            text: firstName + " " + LastName
                            font.pixelSize: 16
                        }

                        Text {
                            text: email
                            font.pixelSize: 12
                            color: "#F7630C"
                        }

                    }

                }

            }
        highlight: Rectangle { color: "#45777777" }
        focus: true

    }

}
